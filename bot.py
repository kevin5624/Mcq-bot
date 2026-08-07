import os
import hashlib
import json
import pypdf
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from supabase import create_client
import google.generativeai as genai
from sentence_transformers import SentenceTransformer

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
GEMINI_KEY = os.environ.get("GEMINI_KEY")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
genai.configure(api_key=GEMINI_KEY)
model_ai = genai.GenerativeModel('gemini-1.5-flash')
embedder = SentenceTransformer('all-MiniLM-L6-v2')

def extract_text_from_pdf(file_path):
    reader = pypdf.PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

def process_with_ai(text):
    prompt = f"""
    Extract questions/notes from text and output ONLY valid JSON array:
    Format:
    [
      {{
        "question": "Question text",
        "option_a": "A", "option_b": "B", "option_c": "C", "option_d": "D",
        "correct_option": "A/B/C/D",
        "explanation": "...",
        "exam": "Exam Name", "subject": "Subject Name", "chapter": "Chapter Name",
        "language": "English"
      }}
    ]
    Rules:
    1. If Subject is Hindi, keep language Hindi. Otherwise translate everything to English.
    2. Convert one-liners/notes into 4-option MCQs.
    
    Text: {text[:8000]}
    """
    response = model_ai.generate_content(prompt)
    try:
        clean_json = response.text.replace("```json", "").replace("```", "").strip()
        return json.loads(clean_json)
    except:
        return []

async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE):
    file = await update.message.document.get_file()
    file_path = f"/tmp/{update.message.document.file_name}"
    await file.download_to_drive(file_path)
    
    await update.message.reply_text("Processing PDF with AI...")
    
    text = extract_text_from_pdf(file_path)
    mcqs = process_with_ai(text)
    
    saved_count = 0
    duplicate_count = 0
    
    for item in mcqs:
        q_text = item.get("question")
        if not q_text:
            continue
            
        q_hash = hashlib.sha256(q_text.encode()).hexdigest()
        vector = embedder.encode(q_text).tolist()
        
        sim_check = supabase.rpc("match_questions", {
            "query_embedding": vector,
            "match_threshold": 0.85,
            "match_count": 1
        }).execute()
        
        if len(sim_check.data) > 0:
            duplicate_count += 1
            continue
            
        try:
            supabase.table("questions").insert({
                "question_text": q_text,
                "option_a": item.get("option_a", ""),
                "option_b": item.get("option_b", ""),
                "option_c": item.get("option_c", ""),
                "option_d": item.get("option_d", ""),
                "correct_option": item.get("correct_option", "A"),
                "explanation": item.get("explanation", ""),
                "exam_name": item.get("exam", "General"),
                "subject_name": item.get("subject", "General"),
                "chapter_name": item.get("chapter", "General"),
                "language": item.get("language", "English"),
                "content_hash": q_hash,
                "embedding": vector
            }).execute()
            saved_count += 1
        except Exception:
            duplicate_count += 1
            
    if os.path.exists(file_path):
        os.remove(file_path)
        
    await update.message.reply_text(f"Processing Complete!\nSaved: {saved_count}\nDuplicates Avoided: {duplicate_count}")

async def stats_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    res = supabase.table("questions").select("id", count="exact").execute()
    await update.message.reply_text(f"Total Questions in Database: {res.count}")

def main():
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    app.add_handler(MessageHandler(filters.Document.ALL, handle_document))
    app.add_handler(CommandHandler("stats", stats_command))
    app.run_polling()

if __name__ == "__main__":
    main()

