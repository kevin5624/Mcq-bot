import os
import re
import json
import time
import base64
import hashlib
import html
import sys
from pathlib import Path

import requests
import pypdf
import pdfplumber
from PIL import Image
from thefuzz import fuzz
from openai import OpenAI

SUPABASE_URL = os.getenv('SUPABASE_URL_1')
SUPABASE_KEY = os.getenv('SUPABASE_KEY_1')
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')

if not SUPABASE_URL or not SUPABASE_KEY or not TELEGRAM_TOKEN:
    raise RuntimeError('SUPABASE_URL_1, SUPABASE_KEY_1 and TELEGRAM_TOKEN are required')

S = requests.Session()
S.headers.update({'apikey': SUPABASE_KEY, 'Authorization': f'Bearer {SUPABASE_KEY}'})


def tg(method, payload):
    r = S.post(f'https://api.telegram.org/bot{TELEGRAM_TOKEN}/{method}', json=payload, timeout=30)
    r.raise_for_status()
    return r.json()


def sb(path, method='GET', payload=None, headers=None, timeout=60):
    h = dict(S.headers)
    if headers:
        h.update(headers)
    r = S.request(method, f'{SUPABASE_URL}/rest/v1/{path}', headers=h, json=payload, timeout=timeout)
    if not r.ok:
        raise RuntimeError(f'Supabase {r.status_code}: {r.text[:1000]}')
    return r.json() if r.text else None


def rpc(name, payload=None):
    return sb(f'rpc/{name}', 'POST', payload or {}, timeout=60)


def claim_job():
    rows = rpc('claim_ingest_job')
    return rows[0] if rows else None


def update_job(job_id, **fields):
    fields['updated_at'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    sb(f'ingest_queue?id=eq.{job_id}', 'PATCH', fields, headers={'Prefer': 'return=minimal'})


def normalize(text):
    return re.sub(r'[^a-z0-9\s]', '', str(text).lower()).strip()


def hash_question(text):
    return hashlib.sha256(normalize(text).encode('utf-8')).hexdigest()


def download_telegram_file(file_id, destination):
    info = tg('getFile', {'file_id': file_id})
    if not info.get('ok'):
        raise RuntimeError(f'Telegram getFile failed: {info}')
    path = info['result']['file_path']
    # Standard Telegram cloud Bot API file download limit is 20 MB.
    r = S.get(f'https://api.telegram.org/file/bot{TELEGRAM_TOKEN}/{path}', timeout=120, stream=True)
    r.raise_for_status()
    with open(destination, 'wb') as f:
        for chunk in r.iter_content(1024 * 1024):
            if chunk:
                f.write(chunk)
    return destination


def extract_pdf(path, pages_per_chunk=5):
    chunks = []
    try:
        reader = pypdf.PdfReader(path)
        total = len(reader.pages)
        buf = []
        for i, page in enumerate(reader.pages, 1):
            buf.append(page.extract_text() or '')
            if i % pages_per_chunk == 0 or i == total:
                text = '\n'.join(buf).strip()
                if len(text) > 30:
                    chunks.append((text, i, total))
                buf = []
        if chunks:
            return chunks
    except Exception as e:
        print('pypdf:', e)
    chunks = []
    with pdfplumber.open(path) as pdf:
        total = len(pdf.pages)
        buf = []
        for i, page in enumerate(pdf.pages, 1):
            buf.append(page.extract_text() or '')
            if i % pages_per_chunk == 0 or i == total:
                text = '\n'.join(buf).strip()
                if len(text) > 30:
                    chunks.append((text, i, total))
                buf = []
    return chunks


def json_array(raw):
    if not raw:
        return []
    raw = raw.strip()
    raw = re.sub(r'^```(?:json)?\s*', '', raw, flags=re.I)
    raw = re.sub(r'\s*```$', '', raw)
    a, b = raw.find('['), raw.rfind(']')
    if a >= 0 and b > a:
        try:
            data = json.loads(raw[a:b+1])
            return data if isinstance(data, list) else []
        except Exception:
            pass
    return []


PROMPT = '''You are a careful educational MCQ parser for Indian competitive-exam/NCERT material.
Extract MCQs and convert useful one-liners/notes into MCQs.

LANGUAGE:
- Database language is English by default. Translate Hindi/general content into clear English.
- If the SUBJECT itself is Hindi or Hindi Literature, keep the question, options and explanation in Hindi.
- Do not preserve Hindi merely because the source PDF is Hindi; decide from the subject.

ONE-LINER/NOTE:
- Put the source's correct fact into option_a and set correct_option to A.
- Create 3 wrong options that are plausible and closely related to the same topic. Never random options.

CATEGORIZE exam, subject and chapter from context. If uncertain, use General Exams / General Knowledge / General.
Difficulty must be Easy, Medium, or Hard.

NCERT CHECK:
Use your knowledge of NCERT Classes 6-12 to correct obvious factual errors. Do not claim that you consulted a live NCERT book. Set ncert_confidence to High/Medium/Low.

Return ONLY a JSON array. Each object must contain:
question, option_a, option_b, option_c, option_d, correct_option, explanation,
difficulty, exam, subject, chapter, language, ncert_confidence.

SOURCE TEXT:
'''


def openai_client(base_url, key):
    if not key:
        return None
    return OpenAI(base_url=base_url, api_key=key)


def call_text_ai(prompt):
    providers = [
        ('Groq', openai_client('https://api.groq.com/openai/v1', os.getenv('GROQ_API_KEY')), os.getenv('GROQ_MODEL', 'llama-3.1-8b-instant')),
        ('OpenRouter', openai_client('https://openrouter.ai/api/v1', os.getenv('OPENROUTER_API_KEY')), os.getenv('OPENROUTER_MODEL', 'meta-llama/llama-3.1-8b-instruct')),
        ('DeepSeek', openai_client('https://api.deepseek.com', os.getenv('DEEPSEEK_API_KEY')), os.getenv('DEEPSEEK_MODEL', 'deepseek-chat')),
        ('Cerebras', openai_client('https://api.cerebras.ai/v1', os.getenv('CEREBRAS_API_KEY')), os.getenv('CEREBRAS_MODEL', 'llama3.1-8b')),
        ('SambaNova', openai_client('https://api.sambanova.ai/v1', os.getenv('SAMBANOVA_API_KEY')), os.getenv('SAMBANOVA_MODEL', 'Meta-Llama-3.1-8B-Instruct')),
        ('Mistral', openai_client('https://api.mistral.ai/v1', os.getenv('MISTRAL_API_KEY')), os.getenv('MISTRAL_MODEL', 'mistral-small-latest')),
        ('Fireworks', openai_client('https://api.fireworks.ai/inference/v1', os.getenv('FIREWORKS_API_KEY')), os.getenv('FIREWORKS_MODEL', 'accounts/fireworks/models/llama-v3p1-8b-instruct')),
        ('DeepInfra', openai_client('https://api.deepinfra.com/v1/openai', os.getenv('DEEPINFRA_API_KEY')), os.getenv('DEEPINFRA_MODEL', 'meta-llama/Meta-Llama-3.1-8B-Instruct')),
        ('Together', openai_client('https://api.together.xyz/v1', os.getenv('TOGETHER_API_KEY')), os.getenv('TOGETHER_MODEL', 'meta-llama/Llama-3.1-8B-Instruct-Turbo')),
    ]
    for name, client, model in providers:
        if not client:
            continue
        try:
            res = client.chat.completions.create(
                model=model,
                messages=[{'role':'user','content':prompt}],
                temperature=0.1,
                max_tokens=6000,
            )
            text = res.choices[0].message.content or ''
            data = json_array(text)
            if data:
                print('AI success:', name)
                return data
        except Exception as e:
            print('AI error', name, str(e)[:500])
    return []


def gemini_vision(path):
    key = os.getenv('GEMINI_KEY')
    if not key:
        return []
    prompt = PROMPT + '\nExtract every visible MCQ from this image. For handwriting, use best-effort OCR.'
    mime = 'image/jpeg'
    suffix = Path(path).suffix.lower()
    if suffix == '.png': mime = 'image/png'
    with open(path, 'rb') as f:
        data = base64.b64encode(f.read()).decode()
    body = {'contents':[{'parts':[{'text':prompt},{'inline_data':{'mime_type':mime,'data':data}}]}]}
    url = f'https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={key}'
    try:
        r = requests.post(url, json=body, timeout=120)
        r.raise_for_status()
        parts = r.json()['candidates'][0]['content']['parts']
        return json_array(''.join(p.get('text','') for p in parts))
    except Exception as e:
        print('Gemini vision error:', e)
        return []


def parse_chunk(text):
    return call_text_ai(PROMPT + text[:14000])


def insert_question(item):
    q = str(item.get('question','')).strip()
    if len(q) < 5:
        return 'invalid'
    h = hash_question(q)
    # Exact duplicate first.
    exact = sb(f'questions?select=id&content_hash=eq.{h}&limit=1')
    if exact:
        return 'duplicate'
    # Semantic-ish trigram duplicate in PostgreSQL.
    try:
        similar = rpc('find_similar_question', {'p_question': q, 'p_threshold': 0.88})
        if similar:
            return 'duplicate'
    except Exception as e:
        print('similarity check:', e)

    subject = str(item.get('subject') or 'General Knowledge').strip()
    language = 'Hindi' if ('hindi' in subject.lower() or 'hindi literature' in subject.lower()) else 'English'
    payload = {
        'question_text': q,
        'option_a': str(item.get('option_a','N/A')),
        'option_b': str(item.get('option_b','N/A')),
        'option_c': str(item.get('option_c','N/A')),
        'option_d': str(item.get('option_d','N/A')),
        'correct_option': str(item.get('correct_option','A')).upper()[:1],
        'explanation': str(item.get('explanation','')),
        'difficulty': str(item.get('difficulty','Medium')).title(),
        'exam_name': str(item.get('exam','General Exams')),
        'subject_name': subject,
        'chapter_name': str(item.get('chapter','General')),
        'language': language,
        'content_hash': h,
        'normalized_question': normalize(q),
    }
    try:
        data = sb('questions', 'POST', payload, headers={'Prefer':'return=representation,resolution=ignore-duplicates'})
        return 'saved' if data else 'duplicate'
    except Exception as e:
        msg = str(e).lower()
        if 'duplicate' in msg or '23505' in msg or 'unique' in msg:
            return 'duplicate'
        print('insert:', e)
        return 'failed'


def notify(chat_id, text):
    try:
        tg('sendMessage', {'chat_id': chat_id, 'text': text})
    except Exception as e:
        print('notify:', e)


def process_file(job):
    job_id, chat_id = job['id'], job['chat_id']
    name = job.get('file_name') or 'upload'
    tmp = Path('/tmp') / f'mcq_{job_id}_{re.sub(r"[^A-Za-z0-9_.-]", "_", name)}'
    try:
        notify(chat_id, f'⚙️ Processing: {name}\n⏳ Starting extraction...')
        download_telegram_file(job['file_id'], str(tmp))
        is_image = (job.get('mime_type','').startswith('image/') or tmp.suffix.lower() in {'.jpg','.jpeg','.png','.webp'})
        if is_image:
            items = gemini_vision(str(tmp))
            chunks = [(items, 1, 1)]
        else:
            chunks_text = extract_pdf(str(tmp))
            chunks = []
            total = len(chunks_text)
            for i, (text, page, pages) in enumerate(chunks_text, 1):
                notify(chat_id, f'⚙️ {name}\nBatch {i}/{total} • page {page}/{pages}\n⏳ AI extraction running...')
                chunks.append((parse_chunk(text), i, total))

        saved = dup = 0
        total_items = sum(len(x[0]) for x in chunks)
        update_job(job_id, total_items=total_items)
        done = 0
        for items, batch, total in chunks:
            for item in items:
                status = insert_question(item)
                if status == 'saved': saved += 1
                elif status == 'duplicate': dup += 1
                done += 1
                if done % 10 == 0 or done == total_items:
                    update_job(job_id, progress=done, saved_count=saved, duplicate_count=dup)
        update_job(job_id, status='done', progress=done, saved_count=saved, duplicate_count=dup, finished_at=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
        notify(chat_id, f'✅ Finished: {name}\n\n📥 Saved: {saved}\n♻️ Duplicates/similar skipped: {dup}\n📊 Parsed: {total_items}')
    except Exception as e:
        print('file error:', repr(e))
        attempts = int(job.get('attempts') or 1)
        if attempts < 3:
            update_job(job_id, status='queued', next_run_at=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(time.time()+60)), error_message=str(e)[:1500])
            notify(chat_id, f'⚠️ Temporary processing error for {name}. Retrying automatically (attempt {attempts}/3).')
        else:
            update_job(job_id, status='failed', error_message=str(e)[:1500], finished_at=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
            notify(chat_id, f'❌ Failed after 3 attempts: {name}\nError: {str(e)[:500]}')
    finally:
        try: tmp.unlink(missing_ok=True)
        except Exception: pass


def trivia_questions(limit=50):
    r = requests.get('https://the-trivia-api.com/v2/questions', params={'limit':limit}, timeout=30)
    r.raise_for_status()
    return r.json()


def scrape_job(job):
    chat_id, job_id = job['chat_id'], job['id']
    target = int(job.get('target_count') or 500)
    notify(chat_id, f'🌐 Open-source scrape started. Target: {target}+ accepted questions.\nThis runs in batches and skips duplicates.')
    saved = dup = checked = 0
    # Pull extra candidates because NCERT-oriented AI review can reject some.
    for round_no in range(1, 31):
        if saved >= target: break
        try:
            raw = trivia_questions(50)
        except Exception as e:
            print('trivia:', e); time.sleep(2); continue
        prompt = '''Review these open-source questions for Indian school/competitive relevance.
Return ONLY a JSON array containing ONLY questions that are factually sound and reasonably compatible with NCERT Classes 6-12 knowledge.
Correct obvious mistakes. If a question is not suitable, omit it. Convert each accepted question into this schema:
question, option_a, option_b, option_c, option_d, correct_option, explanation, difficulty, exam, subject, chapter, language, ncert_confidence.
Keep subject Hindi/Hindi Literature in Hindi; otherwise English.
Never claim live source verification; ncert_confidence is your knowledge-based confidence.
SOURCE JSON:\n''' + json.dumps(raw, ensure_ascii=False)
        items = call_text_ai(prompt)
        checked += len(items)
        for item in items:
            item.setdefault('exam','General Exams')
            item.setdefault('chapter','General')
            st = insert_question(item)
            if st == 'saved': saved += 1
            elif st == 'duplicate': dup += 1
            if saved >= target: break
        update_job(job_id, progress=checked, saved_count=saved, duplicate_count=dup, total_items=checked)
        notify(chat_id, f'🌐 Scrape batch {round_no}\n✅ Saved: {saved}/{target}\n♻️ Skipped: {dup}')
        time.sleep(0.5)
    status = 'done' if saved >= target else 'failed'
    update_job(job_id, status=status, progress=checked, saved_count=saved, duplicate_count=dup, finished_at=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), error_message=None if status=='done' else 'Target not reached within candidate limit')
    notify(chat_id, f'🏁 Scrape finished\n✅ Saved: {saved}\n♻️ Skipped: {dup}\n📚 AI/NCERT-oriented review applied (not live textbook citation).')


def main():
    deadline = time.time() + 15 * 60
    processed = 0
    while time.time() < deadline:
        job = claim_job()
        if not job:
            print('No queued job.')
            break
        processed += 1
        print('Claimed job', job['id'], job.get('job_type'), job.get('file_name'))
        if job.get('job_type') == 'scrape':
            scrape_job(job)
        else:
            process_file(job)
    print('Processed jobs:', processed)
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
