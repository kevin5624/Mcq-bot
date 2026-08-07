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

def split_pdf_by_pages(file_path, pages_per_split=4):
    reader = pypdf.PdfReader(file_path)
    total_pages = len(reader.pages)
    split_files = []
    
    for i in range(0, total_pages, pages_per_split):
        writer = pypdf.PdfWriter()
        end_page = min(i + pages_per_split, total_pages)
        for page_idx in range(i, end_page):
            writer.add_page(reader.pages[page_idx])
            
        temp_chunk_path = f"/tmp/chunk_{i}_{end_page}.pdf"
        with open(temp_chunk_path, "wb") as f:
            writer.write(f)
        split_files.append((temp_chunk_path, i+1, end_page))
        
    return split_files, total_pages

def process_pdf_chunk_with_gemini(chunk_pdf_path):
    prompt = """
    Analyze this PDF pages directly (including two-column layouts, scanned images, and text).
    Extract all questions, MCQs, or study notes/one-liners and return ONLY a raw JSON array of objects.
    
    Each object must have these exact keys:
    "question", "option_a", "option_b", "option_c", "option_d", "correct_option", "explanation", "exam", "subject", "chapter", "language"

    STRICT RULES:
    1. Handle 2-column or multi-column layouts carefully so question reading order is maintained.
    2. If Subject is Hindi, keep language Hindi. Otherwise translate everything to English.
    3. Convert one-liners or notes into proper 4-option MCQs logically.
    4. Return ONLY the raw JSON array. Do not add markdown like ```json or any extra text.
    """
    
    uploaded_file = None
    try:
        # Direct Gemini Native File Upload (OCR & Vision Capable)
        uploaded_file = ai_client.files.upload(file=chunk_pdf_path)
        
        # Wait until file is processed by Gemini
        while uploaded_file.state.name == "PROCESSING":
            time.sleep(1)
            uploaded_file = ai_client.files.get(name=uploaded_file.name)
            
        response = ai_client.models.generate_content(
            model='gemini-1.5-flash',
            contents=[uploaded_file, prompt]
        )
        
        raw_text = response.text.strip()
        if raw_text.startswith("```json"):
            raw_text = raw_text[7:]
        if raw_text.startswith("```"):
            raw_text = raw_text[3:]
        if raw_text.endswith("```"):
            raw_text = raw_text[:-3]
            
        data = json.loads(raw_text.strip())
        
        # Cleanup file from Gemini servers
        ai_client.files.delete(name=uploaded_file.name)
        return data if isinstance(data, list) else []
    except Exception as e:
        print(f"Error processing PDF chunk via OCR Vision: {e}")
        if uploaded_file:
            try:
                ai_client.files.delete(name=uploaded_file.name)
            except Exception:
                pass
        return []

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("👋 Hello! Send me any PDF file (including scanned or 2-column PDFs) and I will extract MCQs to your Supabase Database.")

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
    
    status_msg = await update.message.reply_text("⚙️ Preparing PDF for AI OCR processing...")
    
    try:
        split_chunks, total_pages = split_pdf_by_pages(file_path, pages_per_split=4)
    except Exception as e:
        await status_msg.edit_text(f"❌ Failed to split PDF: {str(e)}")
        if os.path.exists(file_path):
            os.remove(file_path)
        return

    saved_count = 0
    duplicate_count = 0
    total_splits = len(split_chunks)
    
    for idx, (chunk_path, start_p, end_p) in enumerate(split_chunks):
        try:
            await status_msg.edit_text(f"⚙️ Processing Pages {start_p}-{end_p} of {total_pages} (Batch {idx+1}/{total_splits})...\n📥 Saved Questions: {saved_count}")
        except Exception:
            pass
            
        mcqs = process_pdf_chunk_with_gemini(chunk_path)
        
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

        if os.path.exists(chunk_path):
            os.remove(chunk_path)
            
        time.sleep(2)

    if os.path.exists(file_path):
        os.remove(file_path)
        
    await status_msg.edit_text(f"✅ Processing Complete!\n\n📥 Saved Questions: {saved_count}\n⚠️ Duplicates/Skipped: {duplicate_count}")

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
        
