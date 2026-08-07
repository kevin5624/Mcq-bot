import os
import time
import hashlib
import json
import re
import requests
import pypdf
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from supabase import create_client, Client
from groq import Groq
from google import genai
from thefuzz import fuzz

# Environment Variables & API Clients Setup
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GEMINI_KEY = os.environ.get("GEMINI_KEY")

SUPABASE_URL_1 = os.environ.get("SUPABASE_URL_1")
SUPABASE_KEY_1 = os.environ.get("SUPABASE_KEY_1")
SUPABASE_URL_2 = os.environ.get("SUPABASE_URL_2")
SUPABASE_KEY_2 = os.environ.get("SUPABASE_KEY_2")

# Database Clients Cluster
db_clients = []
if SUPABASE_URL_1 and SUPABASE_KEY_1:
    db_clients.append(create_client(SUPABASE_URL_1, SUPABASE_KEY_1))
if SUPABASE_URL_2 and SUPABASE_KEY_2:
    db_clients.append(create_client(SUPABASE_URL_2, SUPABASE_KEY_2))

# AI Clients
groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None
gemini_client = genai.Client(api_key=GEMINI_KEY) if GEMINI_KEY else None


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
    1. Identify 'exam_name', 'subject_name', and 'chapter_name' automatically from the context.
    2. LANGUAGE RULE:
       - If 'subject_name' is Hindi / Hindi Literature, keep the question and options in HINDI.
       - Otherwise, translate everything (Hindi/Other languages) to clear ENGLISH.
    3. ONE-LINERS / NOTES TO MCQ CONVERSION:
       - Place the correct answer in 'option_a' (correct_option="A").
       - Generate 3 WRONG OPTIONS (option_b, option_c, option_d) that are STRICTLY CONTEXTUALLY RELATED to the topic.
    4. NCERT VERIFICATION:
       - Cross-check answers with NCERT Syllabus (Class 6th-12th). Correct any mistakes and state reason in 'explanation'.

    OUTPUT FORMAT (Raw JSON Array ONLY):
    [
      {{
        "question": "Question text here",
        "option_a": "Option A text",
        "option_b": "Option B text",
        "option_c": "Option C text",
        "option_d": "Option D text",
        "correct_option": "A",
        "explanation": "NCERT Verified concept summary",
        "exam": "Exam Name (e.g. UPSC, SSC, General)",
        "subject": "Subject Name (e.g. History, Science, Hindi)",
        "chapter": "Chapter Name",
        "language": "English or Hindi"
      }}
    ]

    Do NOT include markdown like ```json. Output raw JSON array only.

    Text Chunk:
    {text_chunk[:10000]}
    """


def call_ai_with_fallback(text_chunk):
    prompt = build_ai_prompt(text_chunk)
    
    # Provider 1: Groq AI
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
            return parse_json_response(raw)
        except Exception as e:
            print(f"Groq Provider Failed/Limited: {e}. Falling back to Gemini...")

    # Provider 2: Fallback to Gemini AI
    if gemini_client:
        try:
            res = gemini_client.models.generate_content(
                model='gemini-1.5-flash',
                contents=prompt
            )
            raw = res.text.strip()
            return parse_json_response(raw)
        except Exception as e:
            print(f"Gemini Fallback Failed: {e}")

    return []


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


def is_semantic_duplicate(client: Client, question_text: str) -> bool:
    try:
        # Fetch last 30 questions to check similarity fuzzy ratio
        res = client.table("questions").select("question_text").order("id", desc=True).limit(30).execute()
        for row in res.data:
            existing_q = row.get("question_text", "")
            ratio = fuzz.ratio(question_text.lower(), existing_q.lower())
            if ratio > 85:  # 85% similarity threshold
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
    
    subject = str(item.get("subject", "General Awareness")).strip()
    # Enforce Hindi language only if subject is Hindi
    is_hindi_subject = "hindi" in subject.lower()
    language = "Hindi" if is_hindi_subject else "English"

    data_payload = {
        "question_text": q_text_clean,
        "option_a": str(item.get("option_a", "N/A")),
        "option_b": str(item.get("option_b", "N/A")),
        "option_c": str(item.get("option_c", "N/A")),
        "option_d": str(item.get("option_d", "N/A")),
        "correct_option": str(item.get("correct_option", "A")).upper()[:1],
        "explanation": str(item.get("explanation", "NCERT Verified.")),
        "exam_name": str(item.get("exam", "General")),
        "subject_name": subject,
        "chapter_name": str(item.get("chapter", "General")),
        "language": language,
        "content_hash": q_hash
    }

    # Try inserting in primary database, fallback to secondary if needed
    for client in db_clients:
        if is_semantic_duplicate(client, q_text_clean):
            return False, "duplicate"
            
        try:
            client.table("questions").insert(data_payload).execute()
            return True, "saved"
        except Exception:
            # Hash conflict or storage full -> Move to next DB instance in cluster
            continue

    return False, "duplicate_or_failed"


# Automated Open-Source Scraper (Runs via command or schedule)
def fetch_open_source_questions():
    saved = 0
    try:
        url = "https://opentdb.com/api.php?amount=10&type=multiple"
        resp = requests.get(url, timeout=10).json()
        if resp.get("response_code") == 0:
            for item in resp.get("results", []):
                q_item = {
                    "question": item.get("question"),
                    "option_a": item.get("correct_answer"),
                    "option_b": item.get("incorrect_answers")[0] if len(item.get("incorrect_answers")) > 0 else "N/A",
                    "option_c": item.get("incorrect_answers")[1] if len(item.get("incorrect_answers")) > 1 else "N/A",
                    "option_d": item.get("incorrect_answers")[2] if len(item.get("incorrect_answers")) > 2 else "N/A",
                    "correct_option": "A",
                    "explanation": f"Source: OpenTDB Category - {item.get('category')}",
                    "exam": "General Quiz",
                    "subject": item.get("category", "General Knowledge"),
                    "chapter": "Misc"
                }
                status, _ = insert_question_into_cluster(q_item)
                if status:
                    saved += 1
    except Exception as e:
        print(f"Scraper Error: {e}")
    return saved


# Telegram Handlers
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "👋 **Multi-DB MCQ Ingestion Bot Active!**\n\n"
        "📁 Upload any PDF (up to 100MB).\n"
        "📊 Type `/stats` to see detailed category & DB analytics.\n"
        "🌐 Type `/scrape` to auto-fetch open-source questions."
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
        f"🌐 **Total Questions:** `{total_q}`\n\n"
        f"💾 **Database Storage Distribution:**\n" + "\n".join(db_breakdown)
    )
    await update.message.reply_text(response_msg, parse_mode="Markdown")


async def scrape_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = await update.message.reply_text("🌐 Fetching open-source questions...")
    saved = fetch_open_source_questions()
    await msg.edit_text(f"✅ Scraped & Saved `{saved}` new unique open-source questions into Cluster!")


async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE):
    doc = update.message.document
    if doc.file_size > 100 * 1024 * 1024:
        await update.message.reply_text("❌ File exceeds maximum limit of 100MB.")
        return

    file = await doc.get_file()
    file_path = f"/tmp/{doc.file_name}"
    await file.download_to_drive(file_path)
    
    status_msg = await update.message.reply_text("⚙️ **Reading PDF & Rotating AI Providers...**")
    
    try:
        chunks, total_pages = extract_text_chunks_large_pdf(file_path, pages_per_chunk=3)
    except Exception as e:
        await status_msg.edit_text(f"❌ Failed to extract PDF: {str(e)}")
        if os.path.exists(file_path): os.remove(file_path)
        return

    saved_count = 0
    duplicate_count = 0
    total_splits = len(chunks)

    for idx, (chunk_text, page_num) in enumerate(chunks):
        try:
            await status_msg.edit_text(
                f"⚙️ **Processing Batch {idx+1}/{total_splits} (Page {page_num}/{total_pages})**\n"
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

    await status_msg.edit_text(
        f"✅ **PDF Processing Complete!**\n\n"
        f"📥 **NCERT Verified Questions Saved:** `{saved_count}`\n"
        f"⚠️ **Duplicates/Skipped Avoided:** `{duplicate_count}`"
    )


def main():
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("stats", stats_command))
    app.add_handler(CommandHandler("scrape", scrape_command))
    app.add_handler(MessageHandler(filters.Document.ALL, handle_document))
    
    print("Bot Cluster Running...")
    app.run_polling()

if __name__ == "__main__":
    main()
    
