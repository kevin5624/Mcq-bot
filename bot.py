import os
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

def extract_chunks_from_pdf(file_path, pages_per_chunk=5):
    reader = pypdf.PdfReader(file_path)
    total_pages = len(reader.pages)
    chunks = []
    
    current_text = ""
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        current_text += text + "\n"
        if (i + 1) % pages_per_chunk == 0 or (i + 1) == total_pages:
            if len(current_text.strip()) > 30:
                chunks.append(current_text)
            current_text = ""
            
    return chunks

def process_chunk_with_ai(text_chunk):
    prompt = f"""
    Extract all questions, MCQs, or study notes/one-liners from this text and return ONLY a raw JSON array of objects.
    
    Each object must have these keys:
    "question", "option_a", "option_b", "option_c", "option_d", "correct_option", "explanation", "exam", "subject", "chapter", "language"

    RULES:
    1. If Subject is Hindi, keep language Hindi. Otherwise translate everything to English.
    2. Convert notes/one-liners into proper 4-option MCQs.
    3. Output ONLY the JSON array without backticks or ```json.

    Text:
    {text_chunk[:10000]}
    """
    try:
        response = ai_client.models.generate_content(
            model='gemini-1.5-flash',
            contents=prompt
        )
        raw_text = response.text.strip()
        
        # Clean potential markdown formatting
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
    await update.message.reply_text("👋 Hello! Send me any PDF file and I will extract and save MCQs to your Supabase Database.")

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
    
    status_msg = await update.message.reply_text("⚙️ Reading PDF and extracting text...")
    
    chunks = extract_chunks_from_pdf(file_path, pages_per_chunk=5)
    
    if not chunks:
        await status_msg.edit_text("❌ Could not read text from this PDF. If it is an image/scanned PDF, please try a text-based PDF.")
        if os.path.exists(file_path):
            os.remove(file_path)
        return

    saved_count = 0
    duplicate_count = 0
    total_chunks = len(chunks)
    
    for idx, chunk in enumerate(chunks):
        await status_msg.edit_text(f"⚙️ Extracting MCQs from Batch {idx+1}/{total_chunks}...")
        mcqs = process_chunk_with_ai(chunk)
        
        for item in mcqs:
            q_text = item.get("question")
            if not q_text or len(q_text.strip()) < 5:
                continue
                
            q_hash = hashlib.sha256(q_text.strip().encode()).hexdigest()
            
            try:
                supabase.table("questions").insert({
                    "question_text": q_text.strip(),
                    "option_a": item.get("option_a", "N/A"),
                    "option_b": item.get("option_b", "N/A"),
                    "option_c": item.get("option_c", "N/A"),
                    "option_d": item.get("option_d", "N/A"),
                    "correct_option": str(item.get("correct_option", "A")).upper()[:1],
                    "explanation": item.get("explanation", ""),
                    "exam_name": item.get("exam", "General"),
                    "subject_name": item.get("subject", "General"),
                    "chapter_name": item.get("chapter", "General"),
                    "language": item.get("language", "English"),
                    "content_hash": q_hash
                }).execute()
                saved_count += 1
            except Exception:
                duplicate_count += 1

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
        
