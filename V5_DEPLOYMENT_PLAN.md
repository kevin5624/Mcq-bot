# MCQ Bot V5 — Combined Plan

V5 combines the V3 queue/dedupe/webhook foundation with the V4 NCERT verification/evidence layer.

## Do NOT deploy yet
First run the additive SQL migration:
`sql/002_v5_ncert_evidence.sql`

It does not delete existing questions.

## Intended flow
Telegram -> Webhook/Queue -> File extraction/OCR -> AI extraction/classification
-> exact + similar dedupe -> PENDING -> Official NCERT evidence verification
-> VERIFIED / INCORRECT / NOT_VERIFIABLE -> Supabase

## Important
GitHub Actions must not be used as a permanent Telegram polling server.
Use it for bounded worker/scheduled jobs only.

Before production deployment, test:
1. one text PDF
2. one scanned/image PDF
3. one Hindi-subject file
4. one general Hindi file
5. duplicate upload
6. similar question
7. one NCERT-verifiable question
8. one not-verifiable question
