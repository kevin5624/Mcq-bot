import os
import re
import json
import time
import html
import hashlib
import asyncio
import tempfile
from pathlib import Path

import pypdf
import pdfplumber
import requests
from PIL import Image
import google.generativeai as genai
import cohere
from groq import Groq
from together import Together
from huggingface_hub import InferenceClient
from openai import OpenAI
from anthropic import Anthropic

from thefuzz import fuzz
from supabase import create_client

from telegram import Update
from telegram.ext import (
    Application, CommandHandler, MessageHandler,
    ContextTypes, filters
)


# ============================================================
# CONFIG
# ============================================================

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GEMINI_KEY = os.getenv("GEMINI_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
DEEPINFRA_API_KEY = os.getenv("DEEPINFRA_API_KEY")
SAMBANOVA_API_KEY = os.getenv("SAMBANOVA_API_KEY")
CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
FIREWORKS_API_KEY = os.getenv("FIREWORKS_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

SUPABASE_URL_1 = os.getenv("SUPABASE_URL_1")
SUPABASE_KEY_1 = os.getenv("SUPABASE_KEY_1")
SUPABASE_URL_2 = os.getenv("SUPABASE_URL_2")
SUPABASE_KEY_2 = os.getenv("SUPABASE_KEY_2")
SUPABASE_URL_3 = os.getenv("SUPABASE_URL_3")
SUPABASE_KEY_3 = os.getenv("SUPABASE_KEY_3")

MAX_FILE_SIZE = 100 * 1024 * 1024
PDF_PAGES_PER_CHUNK = int(os.getenv("PDF_PAGES_PER_CHUNK", "5"))
MAX_CHUNK_CHARS = int(os.getenv("MAX_CHUNK_CHARS", "12000"))
FUZZY_THRESHOLD = int(os.getenv("FUZZY_THRESHOLD", "92"))

# Gemini model can be changed from GitHub Secrets/Variables later.
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

if not TELEGRAM_TOKEN:
    raise RuntimeError("TELEGRAM_TOKEN is missing.")

db_clients = []
for url, key in [
    (SUPABASE_URL_1, SUPABASE_KEY_1),
    (SUPABASE_URL_2, SUPABASE_KEY_2),
    (SUPABASE_URL_3, SUPABASE_KEY_3),
]:
    if url and key:
        db_clients.append(create_client(url, key))

if not db_clients:
    raise RuntimeError("At least one Supabase URL/KEY pair is required.")

if GEMINI_KEY:
    genai.configure(api_key=GEMINI_KEY)

groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None
cohere_client = cohere.Client(api_key=COHERE_API_KEY) if COHERE_API_KEY else None
together_client = Together(api_key=TOGETHER_API_KEY) if TOGETHER_API_KEY else None
hf_client = InferenceClient(api_key=HUGGINGFACE_TOKEN) if HUGGINGFACE_TOKEN else None
anthropic_client = Anthropic(api_key=ANTHROPIC_API_KEY) if ANTHROPIC_API_KEY else None

def openai_client(base_url, key):
    return OpenAI(base_url=base_url, api_key=key) if key else None

openrouter_client = openai_client("https://openrouter.ai/api/v1", OPENROUTER_API_KEY)
deepinfra_client = openai_client("https://api.deepinfra.com/v1/openai", DEEPINFRA_API_KEY)
sambanova_client = openai_client("https://api.sambanova.ai/v1", SAMBANOVA_API_KEY)
cerebras_client = openai_client("https://api.cerebras.ai/v1", CEREBRAS_API_KEY)
mistral_client = openai_client("https://api.mistral.ai/v1", MISTRAL_API_KEY)
fireworks_client = openai_client("https://api.fireworks.ai/inference/v1", FIREWORKS_API_KEY)
deepseek_client = openai_client("https://api.deepseek.com", DEEPSEEK_API_KEY)

file_queue = asyncio.Queue()
active_jobs = {}
job_lock = asyncio.Lock()


# ============================================================
# HELPERS
# ============================================================

def now_ms():
    return int(time.time() * 1000)

def utc_now():
    return datetime.now(timezone.utc).isoformat()

def format_time(seconds):
    seconds = max(0, int(seconds))
    mins, secs = divmod(seconds, 60)
    hours, mins = divmod(mins, 60)
    if hours:
        return f"{hours}h {mins}m {secs}s"
    return f"{mins}m {secs}s"

def clean_text(value):
    value = "" if value is None else str(value)
    return re.sub(r"\s+", " ", value).strip()

def question_hash(question, a, b, c, d):
    raw = "||".join(clean_text(x).lower() for x in [question, a, b, c, d])
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

def parse_json_response(raw):
    if not raw:
        return []
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
    text = re.sub(r"\s*```$", "", text)
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, list) else []
    except Exception:
        start, end = text.find("["), text.rfind("]")
        if start >= 0 and end > start:
            try:
                obj = json.loads(text[start:end + 1])
                return obj if isinstance(obj, list) else []
            except Exception:
                return []
    return []

def build_prompt(text):
    return f"""
You are an educational MCQ extraction and normalization engine.

IMPORTANT:
- Return ONLY a JSON array. No markdown.
- Extract ALL usable MCQs from the supplied text.
- Convert one-liners, factual notes, and answer statements into MCQs when possible.
- For converted questions, put the source answer among A-D and create 3 plausible,
  topic-related distractors. Do NOT invent unrelated distractors.
- Determine exam, subject, chapter, topic and class when supported by context.
- General content written in Hindi must be translated into clear English.
- If the SUBJECT itself is Hindi/Hindi Literature/Hindi Grammar, preserve the
  question, options and explanation in Hindi.
- Do not translate proper nouns unnecessarily.
- Correct obvious source errors only when the correction is strongly supported.
- Difficulty must be Easy, Medium, or Hard.
- Do NOT claim NCERT verification. Set verification_status to PENDING.
- Do not create a question if the source is too incomplete to make a reliable MCQ.

Schema:
[
  {{
    "question": "...",
    "option_a": "...",
    "option_b": "...",
    "option_c": "...",
    "option_d": "...",
    "correct_option": "A",
    "explanation": "...",
    "difficulty": "Easy",
    "exam": "...",
    "subject": "...",
    "chapter": "...",
    "topic": "...",
    "subtopic": "...",
    "class_level": "...",
    "language": "English",
    "question_type": "MCQ"
  }}
]

TEXT:
{text[:MAX_CHUNK_CHARS]}
"""

def build_vision_prompt():
    return """
Extract all readable educational questions/MCQs from this image.
Also convert clear one-liners/notes into MCQs where enough information exists.

Return ONLY a JSON array with:
question, option_a, option_b, option_c, option_d, correct_option,
explanation, difficulty, exam, subject, chapter, topic, subtopic,
class_level, language, question_type.

General Hindi content -> English.
If the subject is Hindi/Hindi Literature/Hindi Grammar -> preserve Hindi.
For generated distractors, keep them contextually related.
Do not claim NCERT verification; use pending status internally.
"""

# ============================================================
# PDF EXTRACTION
# ============================================================

def extract_pdf_text(file_path):
    chunks = []
    total_pages = 0

    try:
        reader = pypdf.PdfReader(file_path)
        total_pages = len(reader.pages)
        current = []

        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            current.append(text)

            if ((i + 1) % PDF_PAGES_PER_CHUNK == 0
                    or i + 1 == total_pages):
                joined = "\n".join(current).strip()
                if len(joined) >= 50:
                    chunks.append((joined, i + 1))
                current = []

        if chunks:
            return chunks, total_pages, False
    except Exception as e:
        print("pypdf failed:", e)

    # Fallback
    chunks = []
    try:
        with pdfplumber.open(file_path) as pdf:
            total_pages = len(pdf.pages)
            current = []

            for i, page in enumerate(pdf.pages):
                text = page.extract_text() or ""
                current.append(text)

                if ((i + 1) % PDF_PAGES_PER_CHUNK == 0
                        or i + 1 == total_pages):
                    joined = "\n".join(current).strip()
                    if len(joined) >= 50:
                        chunks.append((joined, i + 1))
                    current = []

            return chunks, total_pages, False
    except Exception as e:
        print("pdfplumber failed:", e)

    return [], total_pages, True


# ============================================================
# AI CALLS
# ============================================================

def openai_call(client, model, prompt):
    if not client:
        return None
    res = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
    )
    return res.choices[0].message.content

def call_ai_text(prompt):
    providers = [
        ("Groq", groq_client, os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")),
        ("SambaNova", sambanova_client, os.getenv("SAMBANOVA_MODEL", "Meta-Llama-3.1-8B-Instruct")),
        ("Cerebras", cerebras_client, os.getenv("CEREBRAS_MODEL", "llama3.1-8b")),
        ("DeepSeek", deepseek_client, os.getenv("DEEPSEEK_MODEL", "deepseek-chat")),
        ("OpenRouter", openrouter_client, os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")),
        ("DeepInfra", deepinfra_client, os.getenv("DEEPINFRA_MODEL", "meta-llama/Meta-Llama-3.1-8B-Instruct")),
        ("Mistral", mistral_client, os.getenv("MISTRAL_MODEL", "mistral-small-latest")),
        ("Fireworks", fireworks_client, os.getenv("FIREWORKS_MODEL", "accounts/fireworks/models/llama-v3p1-8b-instruct")),
    ]

    for name, client, model in providers:
        if not client:
            continue
        started = now_ms()
        try:
            result = openai_call(client, model, prompt)
            if result:
                data = parse_json_response(result)
                if data:
                    return data, name, model, now_ms() - started
        except Exception as e:
            print(f"{name} error: {e}")

    if groq_client is None and GEMINI_KEY:
        pass

    if GEMINI_KEY:
        started = now_ms()
        try:
            model = genai.GenerativeModel(GEMINI_MODEL)
            result = model.generate_content(prompt)
            data = parse_json_response(getattr(result, "text", ""))
            if data:
                return data, "Gemini", GEMINI_MODEL, now_ms() - started
        except Exception as e:
            print("Gemini error:", e)

    if cohere_client:
        started = now_ms()
        try:
            result = cohere_client.chat(
                message=prompt,
                model=os.getenv("COHERE_MODEL", "command-r-plus")
            )
            data = parse_json_response(getattr(result, "text", ""))
            if data:
                return data, "Cohere", os.getenv("COHERE_MODEL", "command-r-plus"), now_ms() - started
        except Exception as e:
            print("Cohere error:", e)

    if together_client:
        started = now_ms()
        try:
            result = together_client.chat.completions.create(
                model=os.getenv("TOGETHER_MODEL", "meta-llama/Llama-3.1-8B-Instruct-Turbo"),
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
            data = parse_json_response(result.choices[0].message.content)
            if data:
                return data, "Together", os.getenv("TOGETHER_MODEL", "meta-llama/Llama-3.1-8B-Instruct-Turbo"), now_ms() - started
        except Exception as e:
            print("Together error:", e)

    return [], "none", "", 0

def vision_extract(file_path):
    if not GEMINI_KEY:
        return [], "none", "", 0

    started = now_ms()
    try:
        img = Image.open(file_path)
        model = genai.GenerativeModel(GEMINI_MODEL)
        result = model.generate_content([build_vision_prompt(), img])
        data = parse_json_response(getattr(result, "text", ""))
        return data, "GeminiVision", GEMINI_MODEL, now_ms() - started
    except Exception as e:
        print("Vision error:", e)
        return [], "none", "", now_ms() - started


# ============================================================
# VALIDATION / DEDUP
# ============================================================

def normalize_item(item):
    q = clean_text(item.get("question"))
    opts = [clean_text(item.get(f"option_{x}")) for x in "abcd"]
    correct = clean_text(item.get("correct_option", "A")).upper()[:1]

    if len(q) < 5 or any(len(x) < 1 for x in opts):
        return None
    if correct not in "ABCD":
        return None
    if len(set(x.lower() for x in opts)) < 4:
        return None

    subject = clean_text(item.get("subject")) or "General Knowledge"
    language = clean_text(item.get("language"))
    if "hindi" in subject.lower() or "hindi literature" in subject.lower():
        language = "Hindi"
    else:
        language = "English"

    difficulty = clean_text(item.get("difficulty")).title()
    if difficulty not in {"Easy", "Medium", "Hard"}:
        difficulty = "Medium"

    qtype = clean_text(item.get("question_type")) or "MCQ"
    if qtype not in {"MCQ", "ONE_LINER_CONVERTED", "NOTE_CONVERTED", "OCR_MCQ"}:
        qtype = "MCQ"

    return {
        "question": q,
        "option_a": opts[0],
        "option_b": opts[1],
        "option_c": opts[2],
        "option_d": opts[3],
        "correct_option": correct,
        "explanation": clean_text(item.get("explanation")) or "Concept explanation pending.",
        "exam": clean_text(item.get("exam")) or "General Exams",
        "subject": subject,
        "chapter": clean_text(item.get("chapter")) or "General",
        "topic": clean_text(item.get("topic")),
        "subtopic": clean_text(item.get("subtopic")),
        "class_level": clean_text(item.get("class_level")),
        "language": language,
        "question_type": qtype,
    }

def similar_question_exists(client, item):
    """
    Phase 2 lightweight similarity check.
    Exact hash is handled by the database unique index.
    Fuzzy comparison is limited to matching subject/chapter candidates
    so a huge database does not get downloaded on every question.
    Semantic vector search will be added in the next phase.
    """
    subject = item["subject"]
    chapter = item["chapter"]
    q = item["question"].lower()

    try:
        res = (
            client.table("questions")
            .select("question_text")
            .eq("subject_name", subject)
            .eq("chapter_name", chapter)
            .limit(200)
            .execute()
        )
        for row in (res.data or []):
            old_q = clean_text(row.get("question_text")).lower()
            if old_q and fuzz.token_set_ratio(q, old_q) >= FUZZY_THRESHOLD:
                return True
    except Exception as e:
        print("Fuzzy check error:", e)

    return False

def insert_question(item, source_file="", source_page=None,
                    source_type="PDF", ai_provider="", ai_model=""):
    item = normalize_item(item)
    if not item:
        return False, "invalid"

    qhash = question_hash(
        item["question"], item["option_a"], item["option_b"],
        item["option_c"], item["option_d"]
    )

    payload = {
        "question_text": item["question"],
        "option_a": item["option_a"],
        "option_b": item["option_b"],
        "option_c": item["option_c"],
        "option_d": item["option_d"],
        "correct_option": item["correct_option"],
        "explanation": item["explanation"],
        "exam_name": item["exam"],
        "subject_name": item["subject"],
        "chapter_name": item["chapter"],
        "topic_name": item["topic"] or None,
        "subtopic_name": item["subtopic"] or None,
        "class_level": item["class_level"] or None,
        "difficulty": item["difficulty"],
        "language": item["language"],
        "question_type": item["question_type"],
        "source_type": source_type,
        "source_file": source_file or None,
        "source_page": source_page,
        "verification_status": "PENDING",
        "content_hash": qhash,
        "ai_provider": ai_provider or None,
        "ai_model": ai_model or None,
    }

    for client in db_clients:
        try:
            if similar_question_exists(client, item):
                return False, "similar"

            res = client.table("questions").insert(payload).execute()
            if res.data:
                return True, "saved"

        except Exception as e:
            msg = str(e).lower()
            if "23505" in msg or "duplicate" in msg or "unique" in msg:
                return False, "duplicate"
            print("Insert failure:", e)

    return False, "failed"


# ============================================================
# JOB DATABASE
# ============================================================

def create_job(file_id, user_id, chat_id, file_name, file_size, file_type):
    payload = {
        "telegram_file_id": file_id,
        "telegram_user_id": str(user_id),
        "telegram_chat_id": str(chat_id),
        "file_name": file_name,
        "file_size": file_size or 0,
        "file_type": file_type,
        "status": "QUEUED",
    }
    try:
        res = db_clients[0].table("processing_jobs").insert(payload).execute()
        return res.data[0]["id"] if res.data else None
    except Exception as e:
        print("Job create error:", e)
        return None

def update_job(job_id, **fields):
    if not job_id:
        return
    try:
        db_clients[0].table("processing_jobs").update(fields).eq("id", job_id).execute()
    except Exception as e:
        print("Job update error:", e)

def log_job(job_id, event, message="", level="INFO",
            provider=None, model=None, duration_ms=None):
    if not job_id:
        return
    try:
        db_clients[0].table("processing_logs").insert({
            "job_id": job_id,
            "level": level,
            "event": event,
            "message": message,
            "ai_provider": provider,
            "ai_model": model,
            "duration_ms": duration_ms,
        }).execute()
    except Exception as e:
        print("Log error:", e)


# ============================================================
# FILE PROCESSING
# ============================================================

async def process_job(update, context, job_id, file_id, file_name, file_size):
    start = time.time()
    saved = duplicate = similar = rejected = extracted = 0
    path = None

    await update_job(job_id, status="PROCESSING", started_at=utc_now())

    try:
        tg_file = await context.bot.get_file(file_id)

        suffix = Path(file_name).suffix.lower() or ".bin"
        fd, path = tempfile.mkstemp(prefix="mcq_", suffix=suffix)
        os.close(fd)

        await tg_file.download_to_drive(path)

        is_image = suffix in {".jpg", ".jpeg", ".png", ".webp"}

        if is_image:
            await update_job(job_id, total_batches=1)
            await safe_status(update, f"🖼️ Processing `{file_name}` via Vision OCR...")
            mcqs, provider, model, duration = await asyncio.to_thread(
                vision_extract, path
            )
            extracted += len(mcqs)

            for item in mcqs:
                ok, flag = await asyncio.to_thread(
                    insert_question, item, file_name, None,
                    "IMAGE_OCR", provider, model
                )
                if ok:
                    saved += 1
                elif flag == "duplicate":
                    duplicate += 1
                elif flag == "similar":
                    similar += 1
                else:
                    rejected += 1

            update_job(
                job_id,
                total_batches=1,
                processed_batches=1,
                extracted_questions=extracted,
                saved_questions=saved,
                duplicate_questions=duplicate + similar,
                rejected_questions=rejected,
                progress_percent=100
            )

        else:
            chunks, total_pages, extraction_failed = await asyncio.to_thread(
                extract_pdf_text, path
            )

            if extraction_failed or not chunks:
                # Scanned PDF OCR is intentionally handled in Phase 3.
                await update_job(
                    job_id, status="FAILED",
                    error_message="No selectable PDF text found. Scanned-PDF OCR will be enabled in Phase 3."
                )
                await safe_status(
                    update,
                    "⚠️ PDF me selectable text nahi mila.\n"
                    "Scanned-PDF OCR Phase 3 me add hoga."
                )
                return

            total_batches = len(chunks)
            await update_job(
                job_id,
                total_pages=total_pages,
                total_batches=total_batches
            )

            for idx, (text, page_num) in enumerate(chunks, start=1):
                elapsed = time.time() - start
                avg = elapsed / max(1, idx - 1)
                remaining = avg * (total_batches - idx) if idx > 1 else 0
                percent = (idx / total_batches) * 100

                await safe_status(
                    update,
                    f"⚙️ Processing `{file_name}`\n\n"
                    f"Batch: {idx}/{total_batches}\n"
                    f"Progress: {percent:.0f}%\n"
                    f"Saved: {saved}\n"
                    f"Duplicates/Similar: {duplicate + similar}\n"
                    f"Rejected: {rejected}\n"
                    f"⏱️ Elapsed: {format_time(elapsed)}\n"
                    f"⏳ ETA: {format_time(remaining)}"
                )

                prompt = build_prompt(text)
                mcqs, provider, model, duration = await asyncio.to_thread(
                    call_ai_text, prompt
                )
                extracted += len(mcqs)

                log_job(
                    job_id, "AI_BATCH",
                    f"Batch {idx}/{total_batches}: {len(mcqs)} items",
                    provider=provider, model=model,
                    duration_ms=duration
                )

                for item in mcqs:
                    ok, flag = await asyncio.to_thread(
                        insert_question, item, file_name, page_num,
                        "PDF", provider, model
                    )
                    if ok:
                        saved += 1
                    elif flag == "duplicate":
                        duplicate += 1
                    elif flag == "similar":
                        similar += 1
                    else:
                        rejected += 1

                await update_job(
                    job_id,
                    processed_batches=idx,
                    processed_pages=min(total_pages, page_num),
                    extracted_questions=extracted,
                    saved_questions=saved,
                    duplicate_questions=duplicate + similar,
                    rejected_questions=rejected,
                    progress_percent=percent
                )

        elapsed = time.time() - start
        await update_job(
            job_id,
            status="COMPLETED",
            completed_at=utc_now(),
            extracted_questions=extracted,
            saved_questions=saved,
            duplicate_questions=duplicate + similar,
            rejected_questions=rejected,
            progress_percent=100
        )

        await safe_status(
            update,
            f"✅ Finished: `{file_name}`\n\n"
            f"📥 Extracted: `{extracted}`\n"
            f"💾 Saved: `{saved}`\n"
            f"♻️ Duplicate/Similar skipped: `{duplicate + similar}`\n"
            f"❌ Rejected: `{rejected}`\n"
            f"⏱️ Time: `{format_time(elapsed)}`"
        )

    except Exception as e:
        print("Job error:", e)
        update_job(
            job_id,
            status="FAILED",
            error_message=str(e)[:2000]
        )
        await safe_status(update, f"❌ Processing failed: `{str(e)[:500]}`")

    finally:
        if path:
            try:
                os.remove(path)
            except OSError:
                pass


async def safe_status(update, text):
    try:
        if update.message:
            # Context is shared by this queued job.
            old = update._mcq_status_message if hasattr(update, "_mcq_status_message") else None
            if old:
                await old.edit_text(text)
            else:
                msg = await update.message.reply_text(text)
                try:
                    update._mcq_status_message = msg
                except Exception:
                    pass
    except Exception:
        pass


async def queue_worker():
    while True:
        item = await file_queue.get()
        try:
            await process_job(*item)
        except Exception as e:
            print("Worker error:", e)
        finally:
            file_queue.task_done()


# ============================================================
# TELEGRAM COMMANDS
# ============================================================

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "👋 MCQ Database Engine active.\n\n"
        "📄 PDF/Image upload karo.\n"
        "📊 /stats — database statistics\n"
        "📋 /queue — processing queue\n"
        "🌐 /scrape — open-source scraper Phase 4 me enable hoga."
    )

async def stats_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        total = db_clients[0].table("questions").select(
            "id", count="exact"
        ).execute().count or 0

        exams = db_clients[0].table("questions").select(
            "exam_name", count="exact"
        ).execute()

        await update.message.reply_text(
            f"📊 DATABASE STATUS\n\n"
            f"Total questions: {total}\n"
            f"Supabase databases connected: {len(db_clients)}\n\n"
            f"Use Supabase category views for detailed exam/subject/chapter stats."
        )
    except Exception as e:
        await update.message.reply_text(f"❌ Stats error: {str(e)[:500]}")

async def queue_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        f"📋 Queue pending: {file_queue.qsize()} file(s)"
    )

async def handle_file(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message = update.message

    doc = message.document
    photo = message.photo[-1] if message.photo else None

    if not doc and not photo:
        return

    if doc:
        file_id = doc.file_id
        file_name = doc.file_name or f"document_{int(time.time())}"
        file_size = doc.file_size or 0
        file_type = "DOCUMENT"
    else:
        file_id = photo.file_id
        file_name = f"photo_{int(time.time())}.jpg"
        file_size = photo.file_size or 0
        file_type = "IMAGE"

    if file_size > MAX_FILE_SIZE:
        await message.reply_text(
            "❌ File 100 MB application limit se bada hai."
        )
        return

    job_id = await asyncio.to_thread(
        create_job,
        file_id,
        update.effective_user.id,
        update.effective_chat.id,
        file_name,
        file_size,
        file_type
    )

    await file_queue.put(
        (update, context, job_id, file_id, file_name, file_size)
    )

    position = file_queue.qsize()
    await message.reply_text(
        f"📥 `{file_name}` queue me add ho gaya.\n"
        f"Position: #{position}"
    )


# ============================================================
# MAIN
# ============================================================

async def post_init(application):
    application.create_task(queue_worker())

def main():
    app = (
        Application.builder()
        .token(TELEGRAM_TOKEN)
        .post_init(post_init)
        .build()
    )

    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("stats", stats_command))
    app.add_handler(CommandHandler("queue", queue_command))

    app.add_handler(
        MessageHandler(
            filters.Document.ALL | filters.PHOTO,
            handle_file
        )
    )

    print("MCQ Cluster Engine Phase 2 running...")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
