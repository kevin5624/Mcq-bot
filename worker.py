import os
import re
import json
import time
import base64
import hashlib
import io
from pathlib import Path
from datetime import datetime, timezone
from urllib.parse import urlparse

import requests
import pypdf
import pdfplumber
import fitz
from openai import OpenAI

SUPABASE_URL = os.getenv("SUPABASE_URL_1", "").rstrip("/")
SUPABASE_KEY = os.getenv("SUPABASE_KEY_1", "")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")

MAX_FILE_BYTES = 20 * 1024 * 1024
WORKER_MINUTES = 15
NCERT_TIMEOUT = 25

if not SUPABASE_URL or not SUPABASE_KEY or not TELEGRAM_TOKEN:
    raise RuntimeError("SUPABASE_URL_1, SUPABASE_KEY_1 and TELEGRAM_TOKEN are required")

S = requests.Session()
S.headers.update({
    "apikey": SUPABASE_KEY,
    "Authorization": f"Bearer {SUPABASE_KEY}",
    "Content-Type": "application/json",
})

NCERT_HOSTS = {"ncert.nic.in", "www.ncert.nic.in"}


def now_iso():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def tg(method, payload):
    r = S.post(
        f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/{method}",
        json=payload,
        timeout=30,
    )
    r.raise_for_status()
    data = r.json()
    if not data.get("ok"):
        raise RuntimeError(f"Telegram {method}: {data}")
    return data["result"]


def notify(chat_id, text):
    try:
        tg("sendMessage", {"chat_id": chat_id, "text": text[:4000]})
    except Exception as e:
        print("notify:", e)


def sb(path, method="GET", payload=None, headers=None, timeout=60):
    h = dict(S.headers)
    if headers:
        h.update(headers)
    r = S.request(
        method,
        f"{SUPABASE_URL}/rest/v1/{path}",
        headers=h,
        json=payload,
        timeout=timeout,
    )
    if not r.ok:
        raise RuntimeError(f"Supabase {r.status_code}: {r.text[:1200]}")
    return r.json() if r.text else None


def rpc(name, payload=None):
    return sb(f"rpc/{name}", "POST", payload or {}, timeout=60)


def claim_job():
    rows = rpc("claim_ingest_job")
    return rows[0] if rows else None


def update_job(job_id, **fields):
    fields["updated_at"] = now_iso()
    sb(
        f"ingest_queue?id=eq.{job_id}",
        "PATCH",
        fields,
        headers={"Prefer": "return=minimal"},
    )


def normalize(text):
    # Unicode-safe normalization so Hindi questions do not collapse to an empty hash.
    return re.sub(r"[^\w\s]", "", str(text or "").lower(), flags=re.UNICODE).strip()


def hash_question(text):
    return hashlib.sha256(normalize(text).encode("utf-8")).hexdigest()


def json_array(raw):
    if not raw:
        return []
    raw = str(raw).strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.I)
    raw = re.sub(r"\s*```$", "", raw)
    a, b = raw.find("["), raw.rfind("]")
    if a < 0 or b <= a:
        return []
    try:
        data = json.loads(raw[a:b + 1])
        return data if isinstance(data, list) else []
    except Exception:
        return []


def download_telegram_file(file_id, destination):
    info = tg("getFile", {"file_id": file_id})
    path = info["result"]["file_path"]

    r = S.get(
        f"https://api.telegram.org/file/bot{TELEGRAM_TOKEN}/{path}",
        timeout=120,
        stream=True,
    )
    r.raise_for_status()

    total = 0
    with open(destination, "wb") as f:
        for chunk in r.iter_content(1024 * 1024):
            if not chunk:
                continue
            total += len(chunk)
            if total > MAX_FILE_BYTES:
                raise RuntimeError("File exceeds 20 MB processing limit")
            f.write(chunk)

    return destination


def extract_pdf_text(path, pages_per_chunk=5):
    chunks = []

    try:
        reader = pypdf.PdfReader(path)
        total = len(reader.pages)
        buf = []
        start = 1

        for i, page in enumerate(reader.pages, 1):
            buf.append(page.extract_text() or "")

            if i % pages_per_chunk == 0 or i == total:
                text = "\n".join(buf).strip()

                if len(text) > 30:
                    chunks.append((text, start, i, total))

                buf = []
                start = i + 1

        if chunks:
            return chunks

    except Exception as e:
        print("pypdf:", e)

    chunks = []

    try:
        with pdfplumber.open(path) as pdf:
            total = len(pdf.pages)
            buf = []
            start = 1

            for i, page in enumerate(pdf.pages, 1):
                buf.append(page.extract_text() or "")

                if i % pages_per_chunk == 0 or i == total:
                    text = "\n".join(buf).strip()

                    if len(text) > 30:
                        chunks.append((text, start, i, total))

                    buf = []
                    start = i + 1

    except Exception as e:
        print("pdfplumber:", e)

    return chunks


def render_pdf_pages(path, max_pages=20):
    images = []
    doc = fitz.open(path)

    try:
        for idx, page in enumerate(doc):
            if idx >= max_pages:
                break

            pix = page.get_pixmap(
                matrix=fitz.Matrix(1.5, 1.5),
                alpha=False,
            )

            out = Path("/tmp") / f"ocr_{os.getpid()}_{idx}.jpg"
            pix.save(str(out))
            images.append((str(out), idx + 1))

    finally:
        doc.close()

    return images


PROMPT = """You are a careful educational MCQ parser for Indian competitive-exam and NCERT material.

Return ONLY a JSON array.

Each object MUST contain:
question, option_a, option_b, option_c, option_d, correct_option,
explanation, difficulty, exam, subject, chapter, language,
ncert_class, ncert_subject, ncert_chapter, ncert_source_title, ncert_source_url, ncert_source_page, ncert_evidence

Rules:
- Never invent a citation or pretend to have opened an NCERT book.
- ncert_source_url must be an official NCERT URL only if a real official URL is supplied
  by the source/context. Otherwise return an empty string.
- ncert_evidence must be empty unless supported by supplied source/context.
- ncert_confidence is High/Medium/Low and is knowledge-based, NOT proof of live verification.
- Difficulty: Easy, Medium, or Hard.
- Keep Hindi/Hindi Literature content in Hindi; otherwise use clear English.
- Correct obvious factual errors cautiously.
- Do not invent facts just to complete an MCQ.
"""


def openai_client(base_url, key):
    return OpenAI(base_url=base_url, api_key=key) if key else None


def call_text_ai(prompt):
    providers = [
        ("Groq", "https://api.groq.com/openai/v1", "GROQ_API_KEY", "GROQ_MODEL", "llama-3.1-8b-instant"),
        ("OpenRouter", "https://openrouter.ai/api/v1", "OPENROUTER_API_KEY", "OPENROUTER_MODEL", "meta-llama/llama-3.1-8b-instruct"),
        ("DeepSeek", "https://api.deepseek.com", "DEEPSEEK_API_KEY", "DEEPSEEK_MODEL", "deepseek-chat"),
        ("Cerebras", "https://api.cerebras.ai/v1", "CEREBRAS_API_KEY", "CEREBRAS_MODEL", "llama3.1-8b"),
        ("SambaNova", "https://api.sambanova.ai/v1", "SAMBANOVA_API_KEY", "SAMBANOVA_MODEL", "Meta-Llama-3.1-8B-Instruct"),
        ("Mistral", "https://api.mistral.ai/v1", "MISTRAL_API_KEY", "MISTRAL_MODEL", "mistral-small-latest"),
        ("Fireworks", "https://api.fireworks.ai/inference/v1", "FIREWORKS_API_KEY", "FIREWORKS_MODEL", "accounts/fireworks/models/llama-v3p1-8b-instruct"),
        ("DeepInfra", "https://api.deepinfra.com/v1/openai", "DEEPINFRA_API_KEY", "DEEPINFRA_MODEL", "meta-llama/Meta-Llama-3.1-8B-Instruct"),
        ("Together", "https://api.together.xyz/v1", "TOGETHER_API_KEY", "TOGETHER_MODEL", "meta-llama/Llama-3.1-8B-Instruct-Turbo"),
    ]

    for name, base, key_name, model_name, default_model in providers:
        client = openai_client(base, os.getenv(key_name))

        if not client:
            continue

        try:
            res = client.chat.completions.create(
                model=os.getenv(model_name, default_model),
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=6000,
            )

            items = json_array(res.choices[0].message.content or "")

            if items:
                print("AI success:", name)
                return items

        except Exception as e:
            print("AI error", name, str(e)[:500])

    return []


def gemini_vision(image_path, extra_prompt=""):
    key = os.getenv("GEMINI_KEY")

    if not key:
        return []

    suffix = Path(image_path).suffix.lower()
    mime = "image/png" if suffix == ".png" else "image/jpeg"

    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("ascii")

    body = {
        "contents": [{
            "parts": [
                {
                    "text": (
                        PROMPT
                        + "\nExtract visible MCQs. Use best-effort OCR.\n"
                        + extra_prompt
                    )
                },
                {
                    "inline_data": {
                        "mime_type": mime,
                        "data": encoded,
                    }
                },
            ]
        }]
    }

    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        "gemini-2.0-flash:generateContent"
    )

    try:
        r = requests.post(
            url,
            params={"key": key},
            json=body,
            timeout=120,
        )
        r.raise_for_status()

        parts = r.json()["candidates"][0]["content"]["parts"]
        return json_array("".join(p.get("text", "") for p in parts))

    except Exception as e:
        print("Gemini vision error:", e)
        return []


def parse_text(text):
    return call_text_ai(
        PROMPT
        + "\nSOURCE TEXT:\n"
        + text[:14000]
    )


def parse_pdf(path):
    text_chunks = extract_pdf_text(path)

    if text_chunks:
        output = []

        for text, start, end, total in text_chunks:
            print(f"Text batch pages {start}-{end}/{total}")
            output.append((parse_text(text), start, end))

        return output

    # Scanned/image-only PDF fallback.
    output = []

    for image_path, page_no in render_pdf_pages(path):
        try:
            output.append(
                (
                    gemini_vision(
                        image_path,
                        f"PDF page number: {page_no}",
                    ),
                    page_no,
                    page_no,
                )
            )
        finally:
            try:
                Path(image_path).unlink(missing_ok=True)
            except Exception:
                pass

    return output


def validate_ncert_url(url):
    if not url:
        return False

    try:
        parsed = urlparse(url)
        return (
            parsed.scheme == "https"
            and parsed.hostname in NCERT_HOSTS
        )
    except Exception:
        return False


def fetch_official_ncert_evidence(url, expected_fact):
    """A source becomes VERIFIED only after an official NCERT URL is fetched.

    This intentionally does NOT use an AI claim as proof.
    """

    if not validate_ncert_url(url):
        return {
            "status": "NEEDS_REVIEW",
            "url": "",
            "evidence": "",
            "confidence": "Low",
        }

    try:
        r = requests.get(
            url,
            timeout=NCERT_TIMEOUT,
            headers={
                "User-Agent": "MCQ-Evidence-Checker/5.0"
            },
        )
        r.raise_for_status()

        content_type = r.headers.get("content-type", "").lower()

        if (
            "pdf" in content_type
            or url.lower().split("?")[0].endswith(".pdf")
        ):
            reader = pypdf.PdfReader(io.BytesIO(r.content))
            text = "\n".join(
                page.extract_text() or ""
                for page in reader.pages
            )
        else:
            text = re.sub(r"<[^>]+>", " ", r.text)

        text = re.sub(r"\s+", " ", text).strip()

        if not text:
            return {
                "status": "NEEDS_REVIEW",
                "url": url,
                "evidence": "",
                "confidence": "Low",
            }

        expected = set(
            re.findall(
                r"[\w]{4,}",
                normalize(expected_fact),
                flags=re.UNICODE,
            )
        )

        source = set(
            re.findall(
                r"[\w]{4,}",
                normalize(text[:200000]),
                flags=re.UNICODE,
            )
        )

        overlap = len(expected & source) / max(1, len(expected))

        if overlap >= 0.45:
            return {
                "status": "VERIFIED",
                "url": url,
                "evidence": text[:1200],
                "confidence": (
                    "High" if overlap >= 0.65 else "Medium"
                ),
            }

        return {
            "status": "NEEDS_REVIEW",
            "url": url,
            "evidence": text[:800],
            "confidence": "Low",
        }

    except Exception as e:
        print("NCERT fetch:", e)

        return {
            "status": "NEEDS_REVIEW",
            "url": url,
            "evidence": "",
            "confidence": "Low",
        }


def verify_ncert(item):
    url = str(item.get("ncert_source_url") or "").strip()
    ai_evidence = str(item.get("ncert_evidence") or "").strip()

    question = str(item.get("question") or "").strip()
    explanation = str(item.get("explanation") or "").strip()

    if not validate_ncert_url(url):
        return {
            "status": "NEEDS_REVIEW",
            "source_url": "",
            "evidence": ai_evidence,
            "confidence": str(
                item.get("ncert_confidence") or "Low"
            ),
        }

    result = fetch_official_ncert_evidence(
        url,
        question + " " + explanation,
    )

    if result["status"] == "VERIFIED" and ai_evidence:
        result["evidence"] = (
            ai_evidence
            + "\n\nFetched NCERT text:\n"
            + result["evidence"]
        )

    return {
        "status": result["status"],
        "source_url": result["url"],
        "evidence": result["evidence"],
        "confidence": result["confidence"],
    }


def insert_question(item):
    question = str(item.get("question") or "").strip()

    if len(question) < 5:
        return "invalid"

    content_hash = hash_question(question)

    try:
        exact = sb(
            f"questions?select=id&content_hash=eq.{content_hash}&limit=1"
        )

        if exact:
            return "duplicate"

    except Exception as e:
        print("exact duplicate check:", e)

    try:
        similar = rpc(
            "find_similar_question",
            {
                "p_question": question,
                "p_threshold": 0.88,
            },
        )

        if similar:
            return "duplicate"

    except Exception as e:
        print("similarity check:", e)

    subject = str(
        item.get("subject") or "General Knowledge"
    ).strip()

    language = str(
        item.get("language") or "English"
    )

    if (
        "hindi" in subject.lower()
        or "hindi literature" in subject.lower()
    ):
        language = "Hindi"

    evidence = verify_ncert(item)

    payload = {
        "question_text": question,
        "option_a": str(item.get("option_a") or "N/A"),
        "option_b": str(item.get("option_b") or "N/A"),
        "option_c": str(item.get("option_c") or "N/A"),
        "option_d": str(item.get("option_d") or "N/A"),
        "correct_option": str(
            item.get("correct_option") or "A"
        ).upper()[:1],
        "explanation": str(
            item.get("explanation") or ""
        ),
        "difficulty": str(
            item.get("difficulty") or "Medium"
        ).title(),
        "exam_name": str(
            item.get("exam") or "General Exams"
        ),
        "subject_name": subject,
        "chapter_name": str(
            item.get("chapter") or "General"
        ),
        "language": language,
        "content_hash": content_hash,
        "normalized_question": normalize(question),

        # Exact column names from 002_v5_ncert_evidence.sql.
        "verification_status": evidence["status"],
        "verified_answer": str(
            item.get("verified_answer")
            or item.get("correct_option")
            or ""
        ),
        "verification_reason": (
            "Official NCERT URL fetched and supporting text matched."
            if evidence["status"] == "VERIFIED"
            else "Official NCERT evidence was not sufficiently verified."
        ),
        "ncert_class": str(
            item.get("ncert_class") or ""
        ),
        "ncert_subject": str(
            item.get("ncert_subject") or subject
        ),
        "ncert_chapter": str(
            item.get("ncert_chapter")
            or item.get("chapter")
            or ""
        ),
        "ncert_source_title": str(
            item.get("ncert_source_title") or ""
        ),
        "ncert_source_url": evidence["source_url"],
        "ncert_source_page": str(
            item.get("ncert_source_page") or ""
        ),
        "ncert_source_excerpt": evidence["evidence"],
        "ncert_evidence_hash": (
            hashlib.sha256(
                evidence["evidence"].encode("utf-8")
            ).hexdigest()
            if evidence["evidence"]
            else None
        ),
        "verified_at": (
            now_iso()
            if evidence["status"] == "VERIFIED"
            else None
        ),
        "verification_provider": "official_ncert_fetch",
        "verification_model": "",
    }

    try:
        data = sb(
            "questions",
            "POST",
            payload,
            headers={
                "Prefer": (
                    "return=representation,"
                    "resolution=ignore-duplicates"
                )
            },
        )

        return "saved" if data else "duplicate"

    except Exception as e:
        msg = str(e).lower()

        if (
            "23505" in msg
            or "duplicate" in msg
            or "unique" in msg
        ):
            return "duplicate"

        print("insert:", e)
        return "failed"


def process_file(job):
    job_id = job["id"]
    chat_id = job["chat_id"]
    name = job.get("file_name") or "upload"
    size = int(job.get("file_size") or 0)

    if size > MAX_FILE_BYTES:
        update_job(
            job_id,
            status="failed",
            error_message="File exceeds 20 MB",
        )
        notify(
            chat_id,
            "❌ File is larger than the 20 MB limit.",
        )
        return

    safe_name = re.sub(
        r"[^A-Za-z0-9_.-]",
        "_",
        name,
    )

    tmp = (
        Path("/tmp")
        / f"mcq_v5_{job_id}_{safe_name}"
    )

    try:
        update_job(
            job_id,
            status="processing",
            progress=0,
        )

        notify(
            chat_id,
            f"⚙️ Processing: {name}\n"
            "⏳ Extraction started...",
        )

        download_telegram_file(
            job["file_id"],
            str(tmp),
        )

        mime = str(
            job.get("mime_type") or ""
        ).lower()

        suffix = tmp.suffix.lower()

        if (
            mime.startswith("image/")
            or suffix in {
                ".jpg",
                ".jpeg",
                ".png",
                ".webp",
            }
        ):
            items = gemini_vision(str(tmp))
            chunks = [(items, 1, 1)]

        elif suffix == ".pdf" or "pdf" in mime:
            chunks = parse_pdf(str(tmp))

        else:
            raise RuntimeError(
                "Unsupported file type. "
                "Send a PDF or image."
            )

        total_items = sum(
            len(items)
            for items, _, _ in chunks
        )

        update_job(
            job_id,
            total_items=total_items,
        )

        saved = 0
        duplicates = 0
        failed = 0
        verified = 0
        review = 0
        done = 0

        for items, start_page, end_page in chunks:
            for item in items:
                result = insert_question(item)

                if result == "saved":
                    saved += 1

                    status = verify_ncert(item)["status"]

                    if status == "VERIFIED":
                        verified += 1
                    else:
                        review += 1

                elif result == "duplicate":
                    duplicates += 1

                else:
                    failed += 1

                done += 1

                if (
                    done % 10 == 0
                    or done == total_items
                ):
                    update_job(
                        job_id,
                        progress=done,
                        saved_count=saved,
                        duplicate_count=duplicates,
                    )

        update_job(
            job_id,
            status="done",
            progress=done,
            saved_count=saved,
            duplicate_count=duplicates,
            finished_at=now_iso(),
        )

        notify(
            chat_id,
            f"✅ Finished: {name}\n\n"
            f"📥 Saved: {saved}\n"
            f"♻️ Duplicates: {duplicates}\n"
            f"⚠️ Failed: {failed}\n"
            f"📚 NCERT verified: {verified}\n"
            f"🔎 Needs review: {review}\n"
            f"📊 Parsed: {total_items}",
        )

    except Exception as e:
        print("file error:", repr(e))

        attempts = int(
            job.get("attempts") or 1
        )

        if attempts < 3:
            update_job(
                job_id,
                status="queued",
                next_run_at=time.strftime(
                    "%Y-%m-%dT%H:%M:%SZ",
                    time.gmtime(
                        time.time() + 60
                    ),
                ),
                error_message=str(e)[:1500],
            )

            notify(
                chat_id,
                "⚠️ Temporary error. "
                "Job queued for retry.",
            )

        else:
            update_job(
                job_id,
                status="failed",
                error_message=str(e)[:1500],
                finished_at=now_iso(),
            )

            notify(
                chat_id,
                f"❌ Failed: {name}\n"
                f"Error: {str(e)[:500]}",
            )

    finally:
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass


def trivia_questions(limit=50):
    r = requests.get(
        "https://the-trivia-api.com/v2/questions",
        params={"limit": limit},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()


def scrape_job(job):
    chat_id = job["chat_id"]
    job_id = job["id"]
    target = int(
        job.get("target_count") or 500
    )

    notify(
        chat_id,
        "🌐 Open-source scrape started.\n"
        f"Target: {target}+ accepted questions.\n"
        "AI review is applied; official NCERT "
        "evidence is only marked VERIFIED when "
        "an official NCERT URL is actually fetched.",
    )

    saved = 0
    duplicates = 0
    checked = 0

    for round_no in range(1, 31):
        if saved >= target:
            break

        try:
            raw = trivia_questions(50)

        except Exception as e:
            print("trivia:", e)
            time.sleep(2)
            continue

        prompt = """Review these open-source questions for Indian school/competitive relevance.

Return ONLY a JSON array.

Accept only factually sound questions compatible with NCERT Classes 6-12 knowledge.
Do not invent citations.
If no real official NCERT URL is known from the supplied material, leave
ncert_source_url and ncert_evidence empty.

Schema:
question, option_a, option_b, option_c, option_d, correct_option,
explanation, difficulty, exam, subject, chapter, language,
ncert_class, ncert_subject, ncert_chapter, ncert_source_title, ncert_source_url, ncert_source_page, ncert_evidence

SOURCE JSON:
""" + json.dumps(
            raw,
            ensure_ascii=False,
        )

        items = call_text_ai(prompt)
        checked += len(items)

        for item in items:
            item.setdefault(
                "exam",
                "General Exams",
            )
            item.setdefault(
                "chapter",
                "General",
            )

            status = insert_question(item)

            if status == "saved":
                saved += 1

            elif status == "duplicate":
                duplicates += 1

            if saved >= target:
                break

        update_job(
            job_id,
            progress=checked,
            saved_count=saved,
            duplicate_count=duplicates,
            total_items=checked,
        )

        notify(
            chat_id,
            f"🌐 Scrape batch {round_no}\n"
            f"✅ Saved: {saved}/{target}\n"
            f"♻️ Skipped: {duplicates}",
        )

        time.sleep(0.5)

    status = (
        "done"
        if saved >= target
        else "failed"
    )

    update_job(
        job_id,
        status=status,
        progress=checked,
        saved_count=saved,
        duplicate_count=duplicates,
        finished_at=now_iso(),
        error_message=(
            None
            if status == "done"
            else "Target not reached within candidate limit"
        ),
    )

    notify(
        chat_id,
        "🏁 Scrape finished\n"
        f"✅ Saved: {saved}\n"
        f"♻️ Skipped: {duplicates}\n"
        "📚 AI/NCERT-oriented review applied.\n"
        "🔎 Official NCERT VERIFIED status is only "
        "used after an official NCERT URL is fetched.",
    )


def main():
    deadline = (
        time.time()
        + WORKER_MINUTES * 60
    )

    processed = 0

    while time.time() < deadline:
        job = claim_job()

        if not job:
            print("No queued job.")
            break

        processed += 1

        print(
            "Claimed job:",
            job["id"],
            job.get("job_type"),
            job.get("file_name"),
        )

        if job.get("job_type") == "scrape":
            scrape_job(job)
        else:
            process_file(job)

    print("Processed jobs:", processed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
