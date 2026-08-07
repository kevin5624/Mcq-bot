import os
import time
import hashlib
import json
import pypdf
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from supabase import create_client
from groq import Groq

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
groq_client = Groq(api_key=GROQ_API_KEY)

def extract_chunks_from_pdf(file_path, pages_per_chunk=3):
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

def process_chunk_with_groq(text_chunk):
    prompt = f"""
    Analyze this text extracted from exam PDF/notes.
    It can be MCQs with options OR One-Liner Question & Answer table formats.

    TASK:
    Extract all questions and output ONLY a raw JSON array of objects.

    FORMAT:
    [
      {{
        "question": "Question text here",
        "option_a": "Option A or Correct Answer",
        "option_b": "Option B or Logical distractor",
        "option_c": "Option C or Logical distractor",
        "option_d": "Option D or Logical distractor",
        "correct_option": "A",
        "explanation": "Brief context",
        "exam": "General GK",
        "subject": "General Knowledge",
        "chapter": "General",
        "language": "English"
      }}
    ]

    STRICT INSTRUCTIONS:
    1. If the text is a Question & Answer list (without options), place the correct answer in option_a, generate 3 reasonable fake options for B, C, D, and set correct_option to "A".
    2. If it is already a 4-option MCQ, parse options A, B, C, D and correct answer accurately.
    3. Output ONLY valid JSON array without ```json backticks or extra lines.

    Text:
    {text_chunk[:10000]}
    """
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = groq_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are a JSON parser that outputs valid raw JSON arrays only."},
                    {"role": "user", "content": prompt}
                ],
                model="llama-3.1-8b-instant",
                temperature=0.1
            )
            
            raw_text = response.choices[0].message.content.strip()
            
            if raw_text.startswith("```json"):
                raw_text = raw_text[7:]
            if raw_text.startswith("```"):
                raw_text = raw_text[3:]
            if raw_text.endswith("```"):
                raw_text = raw_text[:-3]
                
            data = json.loads(raw_text.strip())
            return data if isinstance(data, list) else []
            
        except Exception as e:
            err_msg = str(e)
            print(f"Attempt {attempt+1} Groq Error: {err_msg}")
            if "429" in err_msg:
                time.sleep(10)
            else:
                time.sleep(2)
                
    return []

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("👋 Hello! Send me any PDF file (MCQs or One-liner Q&A tables) and I will extract and save them to Supabase.")

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
        await status_msg.edit_text(f"❌ Failed to read PDF: {str(e)}")
        if os.path.exists(file_path):
            os.remove(file_path)
        return

    if not chunks:
        await status_msg.edit_text("❌ Could not extract text from this PDF. Please try a readable text PDF.")
        if os.path.exists(file_path):
            os.remove(file_path)
        return

    saved_count = 0
    duplicate_count = 0
    total_splits = len(chunks)
    
    for idx, (chunk_text, page_num) in enumerate(chunks):
        try:
            await status_msg.edit_text(f"⚙️ Groq AI Processing: Page {page_num}/{total_pages} (Batch {idx+1}/{total_splits})...\n📥 Saved Questions: {saved_count}")
        except Exception:
            pass
            
        mcqs = process_chunk_with_groq(chunk_text)
        
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

        time.sleep(2)

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
            
