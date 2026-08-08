import os
import time
import hashlib
import json
import requests
import asyncio
import html
import pypdf
import cohere
import anthropic
from together import Together
from huggingface_hub import InferenceClient
from openai import OpenAI
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from supabase import create_client, Client
from groq import Groq
from google import genai

# Environment Variables Setup
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")

# Multi-AI API Keys
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GEMINI_KEY = os.environ.get("GEMINI_KEY")
COHERE_API_KEY = os.environ.get("COHERE_API_KEY")
TOGETHER_API_KEY = os.environ.get("TOGETHER_API_KEY")
HUGGINGFACE_TOKEN = os.environ.get("HUGGINGFACE_TOKEN")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
DEEPINFRA_API_KEY = os.environ.get("DEEPINFRA_API_KEY")
SAMBANOVA_API_KEY = os.environ.get("SAMBANOVA_API_KEY")
CEREBRAS_API_KEY = os.environ.get("CEREBRAS_API_KEY")
MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY")
FIREWORKS_API_KEY = os.environ.get("FIREWORKS_API_KEY")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")

# Supabase Cluster Keys
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

# Multi-AI Clients Initialization
groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None
gemini_client = genai.Client(api_key=GEMINI_KEY) if GEMINI_KEY else None
cohere_client = cohere.Client(api_key=COHERE_API_KEY) if COHERE_API_KEY else None
together_client = Together(api_key=TOGETHER_API_KEY) if TOGETHER_API_KEY else None
hf_client = InferenceClient(api_key=HUGGINGFACE_TOKEN) if HUGGINGFACE_TOKEN else None
anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY) if ANTHROPIC_API_KEY else None

openrouter_client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY) if OPENROUTER_API_KEY else None
deepinfra_client = OpenAI(base_url="https://api.deepinfra.com/v1/openai", api_key=DEEPINFRA_API_KEY) if DEEPINFRA_API_KEY else None
sambanova_client = OpenAI(base_url="https://api.sambanova.ai/v1", api_key=SAMBANOVA_API_KEY) if SAMBANOVA_API_KEY else None
cerebras_client = OpenAI(base_url="https://api.cerebras.ai/v1", api_key=CEREBRAS_API_KEY) if CEREBRAS_API_KEY else None
mistral_client = OpenAI(base_url="https://api.mistral.ai/v1", api_key=MISTRAL_API_KEY) if MISTRAL_API_KEY else None
fireworks_client = OpenAI(base_url="https://api.fireworks.ai/inference/v1", api_key=FIREWORKS_API_KEY) if FIREWORKS_API_KEY else None
deepseek_client = OpenAI(base_url="https://api.deepseek.com", api_key=DEEPSEEK_API_KEY) if DEEPSEEK_API_KEY else None

file_queue = asyncio.Queue()


def format_time(seconds):
    mins, secs = divmod(int(seconds), 60)
    return f"{mins}m {secs}s"


def extract_text_chunks_large_pdf(file_path, pages_per_chunk=8):
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
    Extract MCQs, One-Liners, Notes from text below.

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

    OUTPUT FORMAT (Raw JSON Array Only):
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

    Text Chunk:
    {text_chunk[:12000]}
    """


def parse_json_response(raw_text):
    if "```" in raw_text:
        raw_text = raw_text.split("```")[1]
    if raw_text.startswith("json"):
        raw_text = raw_text[4:]
    try:
        data = json.loads(raw_text.strip())
        return data if isinstance(data, list) else []
    except Exception:
        return []


def call_ai_fast(prompt_text):
    prompt = prompt_text if "OUTPUT FORMAT" in prompt_text else build_ai_prompt(prompt_text)
    
    # Priority Fast Rotation Cluster
    providers = [
        ("Groq", lambda: groq_client.chat.completions.create(messages=[{"role": "user", "content": prompt}], model="llama-3.1-8b-instant").choices[0].message.content if groq_client else None),
        ("SambaNova", lambda: sambanova_client.chat.completions.create(messages=[{"role": "user", "content": prompt}], model="Meta-Llama-3.1-8B-Instruct").choices[0].message.content if sambanova_client else None),
        ("Gemini", lambda: gemini_client.models.generate_content(model='gemini-1.5-flash', contents=prompt).text if gemini_client else None),
        ("Cerebras", lambda: cerebras_client.chat.completions.create(messages=[{"role": "user", "content": prompt}], model="llama3.1-8b").choices[0].message.content if cerebras_client else None),
        ("DeepSeek", lambda: deepseek_client.chat.completions.create(messages=[{"role": "user", "content": prompt}], model="deepseek-chat").choices[0].message.content if deepseek_client else None),
        ("Cohere", lambda: cohere_client.chat(message=prompt, model="command-r-plus").text if cohere_client else None),
        ("Claude", lambda: anthropic_client.messages.create(model="claude-3-haiku-20240307", max_tokens=2000, messages=[{"role": "user", "content": prompt}]).content[0].text if anthropic_client else None),
        ("Together", lambda: together_client.chat.completions.create(messages=[{"role": "user", "content": prompt}], model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo").choices[0].message.content if together_client else None),
        ("OpenRouter", lambda: openrouter_client.chat.completions.create(messages=[{"role": "user", "content": prompt}], model="meta-llama/llama-3.3-70b-instruct:free").choices[0].message.content if openrouter_client else None),
        ("DeepInfra", lambda: deepinfra_client.chat.completions.create(messages=[{"role": "user", "content": prompt}], model="meta-llama/Meta-Llama-3.1-8B-Instruct").choices[0].message.content if deepinfra_client else None),
        ("Fireworks", lambda: fireworks_client.chat.completions.create(messages=[{"role": "user", "content": prompt}], model="accounts/fireworks/models/llama-v3p1-8b-instruct").choices[0].message.content if fireworks_client else None),
        ("Mistral", lambda: mistral_client.chat.completions.create(messages=[{"role": "user", "content": prompt}], model="open-mistral-7b").choices[0].message.content if mistral_client else None),
        ("HuggingFace", lambda: hf_client.text_generation(prompt, model="meta-llama/Llama-3.2-3B-Instruct") if hf_client else None)
    ]

    for name, func in providers:
        try:
            res = func()
            if res:
                parsed = parse_json_response(res.strip())
                if parsed:
                    return parsed
        except Exception:
            continue

    time.sleep(2)
    return []


def insert_question_into_cluster(item):
    q_text = item.get("question")
    if not q_text or len(str(q_text).strip()) < 5:
        return False, "invalid"

    q_text_clean = str(q_text).strip()
    q_hash = hashlib.sha256(q_text_clean.lower().encode()).hexdigest()
    
    subject = str(item.get("subject", "General Knowledge")).strip()
    language = "Hindi" if "hindi" in subject.lower() else "English"

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
        try:
            client.table("questions").insert(data_payload).execute()
            return True, "saved"
        except Exception:
            continue

    return False, "duplicate_or_failed"


def fetch_open_source_questions_ncert_verified(target_count=500):
    saved = 0
    batch_size = 50
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
                
                prompt = f"""
                NCERT Verifier: Verify internet MCQs against NCERT facts.
                
                OUTPUT FORMAT (Raw JSON Array):
                [
                  {{
                    "question": "Question text",
                    "option_a": "NCERT Verified Correct Answer",
                    "option_b": "Wrong 1",
                    "option_c": "Wrong 2",
                    "option_d": "Wrong 3",
                    "correct_option": "A",
                    "explanation": "NCERT Note",
                    "exam": "General Exams",
                    "subject": "General Knowledge",
                    "chapter": "Misc"
                  }}
                ]
                
                Data to verify: {json.dumps(raw_questions)}
                """
                
                verified_mcqs = call_ai_fast(prompt)
                for item in verified_mcqs:
                    status, _ = insert_question_into_cluster(item)
                    if status:
                        saved += 1
                        
            time.sleep(1)
            
        except Exception as e:
            print(f"Scraper Batch {loop_idx+1} Error: {e}")
            
    return saved


async def file_queue_worker():
    while True:
        update, context, file_id, file_name, file_size = await file_queue.get()
        start_time = time.time()
        try:
            file = await context.bot.get_file(file_id)
            file_path = f"/tmp/{file_name}"
            await file.download_to_drive(file_path)
            
            status_msg = await update.message.reply_text(f"⚙️ **Processing PDF:** `{file_name}`...")
            
            try:
                chunks, total_pages = extract_text_chunks_large_pdf(file_path, pages_per_chunk=8)
            except Exception as e:
                await status_msg.edit_text(f"❌ Extraction error: {str(e)}")
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

                mcqs = call_ai_fast(chunk_text)
                
                for item in mcqs:
                    status, flag = insert_question_into_cluster(item)
                    if status:
                        saved_count += 1
                    elif flag in ["duplicate", "duplicate_or_failed"]:
                        duplicate_count += 1

            if os.path.exists(file_path): 
                os.remove(file_path)

            total_time = time.time() - start_time
            queue_remaining = file_queue.qsize()
            await status_msg.edit_text(
                f"✅ **Finished Process:** `{file_name}`\n\n"
                f"📥 **NCERT Verified Questions Saved:** `{saved_count}`\n"
                f"⚠️ **Duplicates Skipped:** `{duplicate_count}`\n"
                f"⏱️ **Total Time Taken:** `{format_time(total_time)}`\n"
                f"🔄 **Remaining Queue:** `{queue_remaining}` files"
            )

        except Exception as err:
            print(f"Queue Worker Error: {err}")
        finally:
            file_queue.task_done()


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "👋 **Fast 13-AI Cluster MCQ Bot Active!**\n\n"
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
    msg = await update.message.reply_text("🌐 Starting Fast NCERT Scraper (Target: 500+ Questions)...")
    start_time = time.time()
    saved = fetch_open_source_questions_ncert_verified(target_count=500)
    total_time = time.time() - start_time
    await msg.edit_text(
        f"✅ **Auto-Run Scraper Complete!**\n\n"
        f"📥 **NCERT Verified Questions Saved:** `{saved}`\n"
        f"⏱️ **Total Time Taken:** `{format_time(total_time)}`"
    )


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
    
    print("Fast Bot Engine Running...")
    app.run_polling()

if __name__ == "__main__":
    main()
