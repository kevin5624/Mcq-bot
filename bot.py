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
        text = page.extract_text(layout=True) or "" 
        current_text += text + "\n"
        if (i + 1) % pages_per_chunk == 0 or (i + 1) == total_pages:
            if len(current_text.strip()) > 30:
                chunks.append((current_text, i + 1))
            current_text = ""
            
    return chunks, total_pages

def process_chunk_with_groq_ncert_verify(text_chunk):
    prompt = f"""
    You are an expert Educational Content Creator & NCERT Curriculum Verifier.
    The text below contains MCQs, One-liners, or Question-Answer tables from an exam book/notes.

    TASK:
    1. Extract all questions from the text.
    2. Convert One-Liners or Q&A tables into proper 4-option MCQs.
    3. VERY IMPORTANT (NCERT VERIFICATION): Verify the correct answer against standard NCERT Textbooks (Class 6th to 12th) / Official Syllabus facts. 
       - If the PDF has a wrong answer, CORRECT IT as per NCERT facts.
       - Provide a brief reference or explanation citing NCERT concepts in the 'explanation' field.

    OUTPUT FORMAT (Strict Raw JSON Array ONLY):
    [
      {{
        "question": "Question text",
        "option_a": "Option A text",
        "option_b": "Option B text",
        "option_c": "Option C text",
        "option_d": "Option D text",
        "correct_option": "A/B/C/D",
        "explanation": "Verified with NCERT: Brief explanation of the concept",
        "subject": "Subject Name",
        "exam": "Competitive Exams"
      }}
    ]

    STRICT RULES:
    - Return ONLY the JSON array without backticks (```json) or extra intro/outro text.

    Text to Process:
    {text_chunk[:12000]}
    """
    
    try:
        response = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a JSON parser that strictly verifies exam questions with NCERT textbooks and outputs valid raw JSON arrays only."},
                {"role": "user", "content": prompt}
            ],
            model="llama-3.1-8b-instant",
            temperature=0.1
        )
        
        raw_text = response.choices[0].message.content.strip()
        
        # Markdown cleanup
        if raw_text.startswith("```json"):
            raw_text = raw_text[7:]
        if raw_text.startswith("```"):
            raw_text = raw_text[3:]
        if raw_text.endswith("```"):
            raw_text = raw_text[:-3]
            
        data = json.loads(raw_text.strip())
        return data if isinstance(data, list) else []
        
    except Exception as e:
        print(f"NCERT Verification Error: {e}")
        return []

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("👋 Hello! Send me any PDF (MCQs, One-Liners, or Q&A tables). I will extract, verify answers with NCERT Syllabus, and save them to Supabase Database!")

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
    
    status_msg = await update.message.reply_text("⚙️ Reading PDF & Verifying with NCERT Syllabus...")
    
    try:
        chunks, total_pages = extract_chunks_from_pdf(file_path, pages_per_chunk=3)
    except Exception as e:
        await status_msg.edit_text(f"❌ Failed to read PDF: {str(e)}")
        if os.path.exists(file_path):
            os.remove(file_path)
        return

    saved_count = 0
    duplicate_count = 0
    total_splits = len(chunks)
    
    for idx, (chunk_text, page_num) in enumerate(chunks):
        try:
            await status_msg.edit_text(f"⚙️ NCERT Verification in Progress: Page {page_num}/{total_pages} (Batch {idx+1}/{total_splits})...\n📥 Verified & Saved: {saved_count}")
        except Exception:
            pass
            
        mcqs = process_chunk_with_groq_ncert_verify(chunk_text)
        
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
                    "explanation": str(item.get("explanation", "Verified with NCERT Standards.")),
                    "exam_name": str(item.get("exam", "Competitive Exam")),
                    "subject_name": str(item.get("subject", "General Knowledge")),
                    "content_hash": q_hash
                }).execute()
                saved_count += 1
            except Exception:
                duplicate_count += 1

        time.sleep(1.5)

    if os.path.exists(file_path):
        os.remove(file_path)
        
    await status_msg.edit_text(f"✅ NCERT Verification & Processing Complete!\n\n📥 Verified Questions Saved: {saved_count}\n⚠️ Skipped/Duplicates: {duplicate_count}")

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
        
