# MCQ Bot v3 — Webhook + Queue

This version removes long-running Telegram polling from GitHub Actions.

Flow:
Telegram → Cloudflare Worker webhook → Supabase `ingest_queue` → GitHub Actions worker → AI/OCR → Supabase `questions`

## Important
- Run `sql/001_queue_and_dedupe.sql` in Supabase SQL Editor first.
- Deploy `cloudflare/worker.js` as a Cloudflare Worker.
- Add Cloudflare Worker secrets: `TELEGRAM_TOKEN`, `SUPABASE_URL`, `SUPABASE_SERVICE_ROLE_KEY`.
- Add GitHub secret `ADMIN_CHAT_ID` for the daily scrape notification target (your Telegram chat ID).
- Set the Telegram webhook to the Worker URL.
- Replace the old GitHub workflow with `.github/workflows/worker.yml`.
- Keep existing GitHub API secrets.
- The normal Telegram cloud Bot API download limit is about 20 MB. This version therefore rejects files over 20 MB. A 100 MB requirement needs a self-hosted/local Bot API server and is a separate upgrade.

## What this version does
- webhook instead of `run_polling()`
- persistent Supabase queue
- multiple uploads queued one-by-one
- PDF text extraction
- image OCR via Gemini when available
- AI provider fallback chain
- English translation except Hindi subject/Hindi Literature
- one-liner → MCQ with related distractors
- exact hash + PostgreSQL trigram similarity duplicate filtering
- difficulty tags
- category fields: exam/subject/chapter
- `/start`, `/stats`, `/queue`, `/scrape` commands are handled by the webhook
- daily 500+ scrape jobs can be queued automatically with `ADMIN_CHAT_ID`

## NCERT verification
The worker performs an AI knowledge-based NCERT-oriented review. It does not pretend to consult live NCERT books. A true source-grounded NCERT verifier should be added later using an approved NCERT corpus/reference set.
