import os
import time
import hashlib
import json
import pypdf
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from supabase import create_client
from google import genai

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
GEMINI_KEY = os.environ.get("GEMINI_KEY")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
ai_client = genai.Client(api_key=GEMINI_KEY)

def extract_chunks_from_pdf(file_path, pages_per_chunk=3):
    reader = pypdf.PdfReader(file_path)
    total_pages = len(reader.pages)
    chunks = []
    
    current_text = ""
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        current_text += text + "\n"
        if (i + 1) % pages_per_chunk == 0 or (i + 1) == total_pages:
            chunks.append((current_text, i + 1))
            current_text = ""
            
    return chunks, total_pages

def process_chunk_with_ai(text_chunk):
    prompt = f"""
    Extract all questions, MCQs, or study notes/one-liners from this text and return ONLY a raw JSON array of objects.
    
    Each object must have these exact keys:
    "question", "option_a", "option_b", "option_c", "option_d", "correct_option", "explanation", "exam", "subject", "chapter", "language"

    STRICT RULES:
    1. If Subject is Hindi, keep language Hindi. Otherwise translate everything to English.
    2. Convert notes/one-liners into proper 4-option MCQs.
    3. Output ONLY valid JSON without markdown formatting like ```json.

    Text:
    {text_chunk[:9000]}
    """
    try:
        response = ai_client.models.generate_content(
            model='gemini-1.5-flash',
            contents=prompt
        )
        raw_text = response.text.strip()
        
        if raw_text.startswith("```json"):
            raw_text = raw_text[7:]
        if raw_text.startswith("```"):
            raw_text = raw_text[3:]
        if raw_text.endswith("```"):
            raw_text = raw_text[:-3]
            
        data = json.loads(raw_text.strip())
        return data if isinstance(data, list) else []
    except Exception as e:
        print(f"Error processing chunk: {e}")
        return []

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("👋 Hello! Send me any PDF file and I will extract and save MCQs directly to your Supabase Database.")

async def stats_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        res = supabase.table("questions").select("id", count="exact").execute()
        await update.message.reply_text(f"📊 Total Questions in Database: {res.count}")
    except Exception as e:
        await update.message.reply_text(f"Error fetching stats: {str(e)}")

async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE):
    file = await update.message.document.get_file()
    file_path = f"/tmp/{update.message.document.file_name}"
    await file.download_to_drive(file_path)
    
    status_msg = await update.message.reply_text("⚙️ Reading PDF file...")
    
    try:
        chunks, total_pages = extract_chunks_from_pdf(file_path, pages_per_chunk=3)
    except Exception as e:
        await status_msg.edit_text(f"❌ Error reading PDF: {str(e)}")
        if os.path.exists(file_path):
            os.remove(file_path)
        return

    saved_count = 0
    duplicate_count = 0
    total_chunks = len(chunks)
    
    for idx, (chunk_text, page_num) in enumerate(chunks):
        # Skip if text is empty (scanned PDF image safeguard)
        if len(chunk_text.strip()) < 20:
            continue

        try:
            await status_msg.edit_text(f"⚙️ Processing Batch {idx+1}/{total_chunks} (Page {page_num}/{total_pages})...\n📥 Saved Questions: {saved_count}")
        except Exception:
            pass
            
        mcqs = process_chunk_with_ai(chunk_text)
        
        for item in mcqs:
            q_text = item.get("question")
            if not q_text or len(str(q_text).strip()) < 5:
                continue
                
            q_hash = hashlib.sha256(str(q_text).strip().encode()).hexdigest()
            
            try:
                supabase.table("questions").insert({
                    "question_text": str(q_text).strip(),
                    "option_a": str(item.get("option_a", "N/A")),
                    "option_b": str(item.get("option_b", "N/A")),
                    "option_c": str(item.get("option_c", "N/A")),
                    "option_d": str(item.get("option_d", "N/A")),
                    "correct_option": str(item.get("correct_option", "A")).upper()[:1],
                    "explanation": str(item.get("explanation", "")),
                    "exam_name": str(item.get("exam", "General")),
                    "subject_name": str(item.get("subject", "General")),
                    "chapter_name": str(item.get("chapter", "General")),
                    "language": str(item.get("language", "English")),
                    "content_hash": q_hash
                }).execute()
                saved_count += 1
            except Exception:
                duplicate_count += 1
                
        time.sleep(1.5)

    if os.path.exists(file_path):
        os.remove(file_path)
        
    await status_msg.edit_text(f"✅ Processing Complete!\n\n📥 Total Saved: {saved_count}\n⚠️ Duplicates/Skipped: {duplicate_count}")

def main():
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("stats", stats_command))
    app.add_handler(CommandHandler("status", stats_command))
    app.add_handler(MessageHandler(filters.Document.ALL, handle_document))
    
    print("Bot is running...")
    app.run_polling()

if __name__ == "__main__":
    main()
