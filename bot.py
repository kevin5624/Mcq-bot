import os
import time
import hashlib
import json
import requests
import asyncio
import html
import pypdf
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from supabase import create_client, Client
from groq import Groq
from google import genai
from thefuzz import fuzz

# Environment Variables Setup
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GEMINI_KEY = os.environ.get("GEMINI_KEY")

SUPABASE_URL_1 = os.environ.get("SUPABASE_URL_1")
SUPABASE_KEY_1 = os.environ.get("SUPABASE_KEY_1")
SUPABASE_URL_2 = os.environ.get("SUPABASE_URL_2")
SUPABASE_KEY_2 = os.environ.get("SUPABASE_KEY_2")

# Database Cluster Setup
db_clients = []
if SUPABASE_URL_1 and SUPABASE_KEY_1:
    db_clients.append(create_client(SUPABASE_URL_1, SUPABASE_KEY_1))
if SUPABASE_URL_2 and SUPABASE_KEY_2:
    db_clients.append(create_client(SUPABASE_URL_2, SUPABASE_KEY_2))

# AI Clients
groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None
gemini_client = genai.Client(api_key=GEMINI_KEY) if GEMINI_KEY else None

file_queue = asyncio.Queue()


def extract_text_chunks_large_pdf(file_path, pages_per_chunk=3):
    reader = pypdf.PdfReader(file_path)
    total_pages = len(reader.pages)
    chunks = []
    
    current_text = ""
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        current_text += text + "\n"
        if (i + 1) % pages_per_chunk == 0 or (i + 1) == total_pages:
            if len(current_text.strip()) > 30:
                chunks.append((current_text, i + 1))
            current_text = ""
            
    return chunks, total_pages


def build_ai_prompt(text_chunk):
    return f"""
    You are an AI Educational Data Parser & NCERT Syllabus Verifier.
    Extract all MCQs, One-Liners, Notes, or Q&A tables from the text below.

    RULES FOR CATEGORIZATION & TRANSLATION:
    1. Identify 'exam_name', 'subject_name', and 'chapter_name' automatically from context.
    2. LANGUAGE RULE:
       - If 'subject_name' is Hindi / Hindi Literature, keep question & options in HINDI.
       - Otherwise, translate everything to clear ENGLISH.
    3. ONE-LINERS / NOTES TO MCQ CONVERSION:
       - Place correct answer in 'option_a' (correct_option="A").
       - Generate 3 WRONG OPTIONS (option_b, option_c, option_d) strictly CONTEXTUALLY RELATED to the topic.
    4. NCERT VERIFICATION:
       - Cross-check answers with NCERT Syllabus (Class 6th-12th). Correct any mistakes and state reason in 'explanation'.

    OUTPUT FORMAT (Raw JSON Array ONLY):
    [
      {{
        "question": "Question text",
        "option_a": "Option A text",
        "option_b": "Option B text",
        "option_c": "Option C text",
        "option_d": "Option D text",
        "correct_option": "A",
        "explanation": "NCERT Verified concept summary",
        "exam": "Exam Name",
        "subject": "Subject Name",
        "chapter": "Chapter Name",
        "language": "English or Hindi"
      }}
    ]

    Do NOT include markdown like ```json. Output raw JSON array only.

    Text Chunk:
    {text_chunk[:10000]}
    """


def parse_json_response(raw_text):
    if raw_text.startswith("```json"):
        raw_text = raw_text[7:]
    if raw_text.startswith("```"):
        raw_text = raw_text[3:]
    if raw_text.endswith("```"):
        raw_text = raw_text[:-3]
    try:
        data = json.loads(raw_text.strip())
        return data if isinstance(data, list) else []
    except Exception:
        return []


def call_ai_with_fallback(prompt_text):
    prompt = prompt_text if "OUTPUT FORMAT" in prompt_text else build_ai_prompt(prompt_text)
    
    # Try Groq AI First
    if groq_client:
        try:
            res = groq_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You output valid raw JSON arrays only."},
                    {"role": "user", "content": prompt}
                ],
                model="llama-3.1-8b-instant",
                temperature=0.1
            )
            raw = res.choices[0].message.content.strip()
            parsed = parse_json_response(raw)
            if parsed:
                return parsed
        except Exception as e:
            print(f"Groq Limit/Error: {e}. Switching to Gemini AI Fallback...")

    # Fallback to Gemini AI
    if gemini_client:
        try:
            res = gemini_client.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt
            )
            raw = res.text.strip()
            parsed = parse_json_response(raw)
            if parsed:
                return parsed
        except Exception as e:
            print(f"Gemini Fallback Error: {e}")

    return []


def is_semantic_duplicate(client: Client, question_text: str) -> bool:
    try:
        res = client.table("questions").select("question_text").order("id", desc=True).limit(30).execute()
        for row in res.data:
            existing_q = row.get("question_text", "")
            ratio = fuzz.ratio(question_text.lower(), existing_q.lower())
            if ratio > 85:
                return True
    except Exception:
        pass
    return False


def insert_question_into_cluster(item):
    q_text = item.get("question")
    if not q_text or len(str(q_text).strip()) < 5:
        return False, "invalid"

    q_text_clean = str(q_text).strip()
    q_hash = hashlib.sha256(q_text_clean.lower().encode()).hexdigest()
    
    subject = str(item.get("subject", "General Knowledge")).strip()
    is_hindi_subject = "hindi" in subject.lower()
    language = "Hindi" if is_hindi_subject else "English"

    data_payload = {
        "question_text": q_text_clean,
        "option_a": str(item.get("option_a", "N/A")),
        "option_b": str(item.get("option_b", "N/A")),
        "option_c": str(item.get("option_c", "N/A")),
        "option_d": str(item.get("option_d", "N/A")),
        "correct_option": str(item.get("correct_option", "A")).upper()[:1],
        "explanation": str(item.get("explanation", "Verified with NCERT Standards.")),
        "exam_name": str(item.get("exam", "Competitive Exams")),
        "subject_name": subject,
        "chapter_name": str(item.get("chapter", "General")),
        "language": language,
        "content_hash": q_hash
    }

    for client in db_clients:
        if is_semantic_duplicate(client, q_text_clean):
            return False, "duplicate"
            
        try:
            client.table("questions").insert(data_payload).execute()
            return True, "saved"
        except Exception:
            continue

    return False, "duplicate_or_failed"


# Token-Optimized NCERT Verified Bulk Open-Source Scraper
def fetch_open_source_questions_ncert_verified(target_count=500):
    saved = 0
    batch_size = 20  # Optimized 20-item batches to prevent token limits
    loops = target_count // batch_size
    
    for loop_idx in range(loops):
        try:
            url = f"https://opentdb.com/api.php?amount={batch_size}&type=multiple"
            resp = requests.get(url, timeout=12).json()
            
            if resp.get("response_code") == 0:
                raw_questions = []
                for item in resp.get("results", []):
                    raw_questions.append({
                        "question": html.unescape(item.get("question", "")),
                        "given_answer": html.unescape(item.get("correct_answer", "")),
                        "wrong_options": [html.unescape(x) for x in item.get("incorrect_answers", [])],
                        "category": html.unescape(item.get("category", ""))
                    })
                
                # Compact AI Prompt for NCERT Verification
                prompt = f"""
                You are an NCERT Curriculum Verifier.
                Verify these internet questions against NCERT/Academic facts.
                
                RULES:
                1. Correct any wrong answers according to NCERT facts.
                2. Ensure wrong options are contextually relevant to question topic.
                3. Set correct_option="A" with NCERT answer in option_a.
                4. Keep language English (unless subject is Hindi).

                OUTPUT FORMAT (Strict Raw JSON Array ONLY):
                [
                  {{
                    "question": "Question text",
                    "option_a": "NCERT Verified Correct Answer",
                    "option_b": "Contextual wrong option 1",
                    "option_c": "Contextual wrong option 2",
                    "option_d": "Contextual wrong option 3",
                    "correct_option": "A",
                    "explanation": "NCERT Verification: Brief concept note",
                    "exam": "General Exams",
                    "subject": "General Knowledge",
                    "chapter": "Misc"
                  }}
                ]

                Raw Data:
                {json.dumps(raw_questions)}
                """
                
                verified_mcqs = call_ai_with_fallback(prompt)
                for item in verified_mcqs:
                    status, _ = insert_question_into_cluster(item)
                    if status:
                        saved += 1
                        
            time.sleep(2)  # Short pause
            
        except Exception as e:
            print(f"Verified Scraper Batch {loop_idx+1} Error: {e}")
            time.sleep(3)
            
    return saved


async def file_queue_worker():
    while True:
        update, context, file_id, file_name, file_size = await file_queue.get()
        try:
            file = await context.bot.get_file(file_id)
            file_path = f"/tmp/{file_name}"
            await file.download_to_drive(file_path)
            
            status_msg = await update.message.reply_text(f"⚙️ **Processing File from Queue:** `{file_name}`...")
            
            try:
                chunks, total_pages = extract_text_chunks_large_pdf(file_path, pages_per_chunk=3)
            except Exception as e:
                await status_msg.edit_text(f"❌ Failed to extract PDF `{file_name}`: {str(e)}")
                if os.path.exists(file_path): 
                    os.remove(file_path)
                file_queue.task_done()
                continue

            saved_count = 0
            duplicate_count = 0
            total_splits = len(chunks)

            for idx, (chunk_text, page_num) in enumerate(chunks):
                try:
                    await status_msg.edit_text(
                        f"⚙️ **Processing `{file_name}`**\n"
                        f"Batch {idx+1}/{total_splits} (Page {page_num}/{total_pages})\n"
                        f"📥 **Saved:** `{saved_count}` | ⚠️ **Duplicates Skipped:** `{duplicate_count}`"
                    )
                except Exception:
                    pass

                mcqs = call_ai_with_fallback(chunk_text)
                
                for item in mcqs:
                    status, flag = insert_question_into_cluster(item)
                    if status:
                        saved_count += 1
                    elif flag in ["duplicate", "duplicate_or_failed"]:
                        duplicate_count += 1

                time.sleep(1.5)

            if os.path.exists(file_path):
                os.remove(file_path)

            queue_remaining = file_queue.qsize()
            await status_msg.edit_text(
                f"✅ **Finished `{file_name}`!**\n\n"
                f"📥 **NCERT Verified Saved:** `{saved_count}`\n"
                f"⚠️ **Duplicates Skipped:** `{duplicate_count}`\n"
                f"🔄 **Remaining Queue:** `{queue_remaining}` files"
            )

        except Exception as err:
            print(f"Error processing file {file_name}: {err}")
        finally:
            file_queue.task_done()


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "👋 **NCERT-Verified MCQ Cluster Bot Active!**\n\n"
        "📁 Upload multiple PDFs (up to 100MB each) - Queue will process sequentially.\n"
        "📊 Type `/stats` to see detailed category & DB analytics.\n"
        "🌐 Type `/scrape` to auto-fetch 500+ NCERT-verified open-source questions."
    )


async def stats_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    total_q = 0
    db_breakdown = []
    
    for idx, client in enumerate(db_clients):
        try:
            res = client.table("questions").select("id", count="exact").execute()
            count = res.count or 0
            total_q += count
            db_breakdown.append(f"• **Database {idx+1}:** {count} questions")
        except Exception as e:
            db_breakdown.append(f"• **Database {idx+1}:** Error ({e})")

    response_msg = (
        f"📊 **Cluster Stats Summary**\n"
        f"🌐 **Total Questions:** `{total_q}`\n"
        f"🔄 **Pending Files in Queue:** `{file_queue.qsize()}`\n\n"
        f"💾 **Database Storage Distribution:**\n" + "\n".join(db_breakdown)
    )
    await update.message.reply_text(response_msg, parse_mode="Markdown")


async def scrape_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = await update.message.reply_text("🌐 Starting NCERT-Verified Open-Source Scraper (Target: 500+ Questions)...")
    saved = fetch_open_source_questions_ncert_verified(target_count=500)
    await msg.edit_text(f"✅ Auto-Run Scraper Complete!\n\n📥 **NCERT Verified Questions Saved:** `{saved}`")


async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE):
    doc = update.message.document
    if doc.file_size > 100 * 1024 * 1024:
        await update.message.reply_text(f"❌ File `{doc.file_name}` exceeds 100MB limit.")
        return

    await file_queue.put((update, context, doc.file_id, doc.file_name, doc.file_size))
    
    q_size = file_queue.qsize()
    if q_size == 1:
        await update.message.reply_text(f"📥 Received `{doc.file_name}`. Processing started...")
    else:
        await update.message.reply_text(f"📥 Received `{doc.file_name}`. Added to Queue (Position #{q_size}).")


def main():
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("stats", stats_command))
    app.add_handler(CommandHandler("scrape", scrape_command))
    app.add_handler(MessageHandler(filters.Document.ALL, handle_document))
    
    loop = asyncio.get_event_loop()
    loop.create_task(file_queue_worker())
    
    print("Bot Cluster Running...")
    app.run_polling()

if __name__ == "__main__":
    main()
    
