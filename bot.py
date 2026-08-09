import os
import time
import requests

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
SUPABASE_URL = os.getenv("SUPABASE_URL_1", "").rstrip("/")
SUPABASE_KEY = os.getenv("SUPABASE_KEY_1")

MAX_FILE_BYTES = 20 * 1024 * 1024

if not TELEGRAM_TOKEN:
    raise RuntimeError("TELEGRAM_TOKEN is required")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("SUPABASE_URL_1 and SUPABASE_KEY_1 are required")

TG = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"

SB_HEADERS = {
    "apikey": SUPABASE_KEY,
    "Authorization": f"Bearer {SUPABASE_KEY}",
    "Content-Type": "application/json",
    "Prefer": "return=minimal",
}


def telegram(method, payload=None):
    r = requests.post(
        f"{TG}/{method}",
        json=payload or {},
        timeout=35,
    )
    r.raise_for_status()
    data = r.json()

    if not data.get("ok"):
        raise RuntimeError(data)

    return data["result"]


def send_message(chat_id, text):
    try:
        telegram(
            "sendMessage",
            {
                "chat_id": chat_id,
                "text": text[:4000],
            },
        )
    except Exception as e:
        print("sendMessage error:", e)


def queue_job(
    chat_id,
    file_id,
    file_name,
    mime_type,
    file_size,
):
    payload = {
        "chat_id": int(chat_id),
        "job_type": "file",
        "file_id": file_id,
        "file_name": file_name,
        "mime_type": mime_type,
        "file_size": int(file_size or 0),
    }

    r = requests.post(
        f"{SUPABASE_URL}/rest/v1/ingest_queue",
        headers=SB_HEADERS,
        json=payload,
        timeout=30,
    )

    if not r.ok:
        raise RuntimeError(
            f"Supabase {r.status_code}: {r.text[:1000]}"
        )


def choose_file(message):
    # Telegram document
    if message.get("document"):
        f = message["document"]

        return {
            "file_id": f["file_id"],
            "file_name": f.get("file_name") or "document",
            "mime_type": f.get("mime_type") or "",
            "file_size": int(f.get("file_size") or 0),
        }

    # Telegram photo
    # Telegram provides several sizes; choose the largest.
    if message.get("photo"):
        f = message["photo"][-1]

        return {
            "file_id": f["file_id"],
            "file_name": "telegram_photo.jpg",
            "mime_type": "image/jpeg",
            "file_size": int(f.get("file_size") or 0),
        }

    return None


def process_update(update):
    message = update.get("message")

    if not message:
        return

    chat_id = message["chat"]["id"]

    # /start
    text = message.get("text", "").strip()

    if text == "/start":
        send_message(
            chat_id,
            "🤖 MCQ V5 Bot ready.\n\n"
            "📄 PDF/image भेजें।\n"
            "📦 Maximum file size: 20 MB\n\n"
            "File queue में जाएगी और Worker उसे process करेगा.",
        )
        return

    # /status
    if text == "/status":
        send_message(
            chat_id,
            "📥 Your file will be queued first.\n"
            "⚙️ Worker queue से processing करेगा.",
        )
        return

    file_data = choose_file(message)

    if not file_data:
        # Ignore normal text.
        return

    size = file_data["file_size"]

    # Hard 20 MB limit before queueing.
    if size > MAX_FILE_BYTES:
        send_message(
            chat_id,
            "❌ File rejected.\n\n"
            f"📦 Size: {size / 1024 / 1024:.2f} MB\n"
            "🚫 Maximum allowed: 20 MB",
        )
        return

    try:
        queue_job(
            chat_id=chat_id,
            file_id=file_data["file_id"],
            file_name=file_data["file_name"],
            mime_type=file_data["mime_type"],
            file_size=size,
        )

        send_message(
            chat_id,
            "✅ File received and queued.\n\n"
            f"📄 {file_data['file_name']}\n"
            f"📦 {size / 1024 / 1024:.2f} MB\n"
            "⏳ Worker will process it from the queue.",
        )

    except Exception as e:
        print("Queue error:", repr(e))

        send_message(
            chat_id,
            "⚠️ File queue में add नहीं हो सकी.\n"
            "Please try again later.",
        )


def main():
    print("Telegram V5 Intake starting...")

    # Avoid receiving old updates when the process starts.
    try:
        telegram(
            "deleteWebhook",
            {"drop_pending_updates": False},
        )
    except Exception as e:
        print("deleteWebhook:", e)

    offset = None

    while True:
        try:
            updates = telegram(
                "getUpdates",
                {
                    "timeout": 25,
                    "offset": offset,
                    "allowed_updates": ["message"],
                },
            )

            for update in updates:
                offset = update["update_id"] + 1

                try:
                    process_update(update)
                except Exception as e:
                    print(
                        "Update processing error:",
                        repr(e),
                    )

        except requests.RequestException as e:
            print("Telegram network error:", e)
            time.sleep(5)

        except Exception as e:
            print("Intake error:", repr(e))
            time.sleep(5)


if __name__ == "__main__":
    main()
