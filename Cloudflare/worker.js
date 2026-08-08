// Telegram webhook receiver for MCQ Bot v3.
// Deploy as a Cloudflare Worker. Keep SUPABASE_SERVICE_ROLE_KEY and TELEGRAM_TOKEN as secrets.

async function supabase(path, env, method = 'GET', body = null) {
  const r = await fetch(`${env.SUPABASE_URL}/rest/v1/${path}`, {
    method,
    headers: {
      'apikey': env.SUPABASE_SERVICE_ROLE_KEY,
      'Authorization': `Bearer ${env.SUPABASE_SERVICE_ROLE_KEY}`,
      'Content-Type': 'application/json',
      'Prefer': 'return=representation,resolution=ignore-duplicates'
    },
    body: body == null ? undefined : JSON.stringify(body)
  });
  if (!r.ok) throw new Error(`Supabase ${r.status}: ${await r.text()}`);
  return r.status === 204 ? null : r.json();
}

async function telegram(method, env, body) {
  const r = await fetch(`https://api.telegram.org/bot${env.TELEGRAM_TOKEN}/${method}`, {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(body)
  });
  return r.json();
}

function getFile(update) {
  const m = update.message;
  if (!m) return null;
  if (m.document) {
    return {
      file_id: m.document.file_id,
      file_name: m.document.file_name || `document_${m.message_id}`,
      mime_type: m.document.mime_type || 'application/octet-stream',
      file_size: m.document.file_size || 0
    };
  }
  if (m.photo && m.photo.length) {
    const p = m.photo[m.photo.length - 1];
    return {
      file_id: p.file_id,
      file_name: `photo_${m.message_id}.jpg`,
      mime_type: 'image/jpeg',
      file_size: p.file_size || 0
    };
  }
  return null;
}

async function handleCommand(update, env) {
  const m = update.message;
  if (!m || typeof m.text !== 'string' || !m.text.startsWith('/')) return false;
  const chatId = m.chat.id;
  const cmd = m.text.trim().split(/\s+/)[0].split('@')[0].toLowerCase();

  if (cmd === '/start') {
    await telegram('sendMessage', env, {chat_id: chatId, text: '👋 MCQ Bot v3 active.\n\n📄 Send a PDF or image. Files are queued and processed one-by-one.\n📊 /stats — database count\n⏳ /queue — pending jobs\n🌐 /scrape — queue 500+ open-source candidates'});
    return true;
  }
  if (cmd === '/queue') {
    const rows = await supabase('ingest_queue?select=status&status=in.(queued,processing)', env);
    const queued = (rows || []).filter(x => x.status === 'queued').length;
    const processing = (rows || []).filter(x => x.status === 'processing').length;
    await telegram('sendMessage', env, {chat_id: chatId, text: `⏳ Queue\nQueued: ${queued}\nProcessing: ${processing}`});
    return true;
  }
  if (cmd === '/stats') {
    const r = await fetch(`${env.SUPABASE_URL}/rest/v1/questions?select=id`, {headers:{'apikey':env.SUPABASE_SERVICE_ROLE_KEY,'Authorization':`Bearer ${env.SUPABASE_SERVICE_ROLE_KEY}`,'Prefer':'count=exact,head=true'}});
    const count = Number((r.headers.get('content-range') || '0-0/0').split('/')[1] || 0);
    await telegram('sendMessage', env, {chat_id: chatId, text: `📊 Supabase Questions\nTotal: ${count}`});
    return true;
  }
  if (cmd === '/scrape') {
    await supabase('ingest_queue', env, 'POST', {telegram_update_id: update.update_id, chat_id: chatId, job_type:'scrape', target_count:500, file_id:null, file_name:null, mime_type:null, file_size:0});
    await telegram('sendMessage', env, {chat_id: chatId, text:'🌐 500+ open-source scrape job queued. It will run automatically.'});
    return true;
  }
  return false;
}

export default {
  async fetch(request, env) {
    if (request.method !== 'POST') return new Response('OK', {status: 200});
    const update = await request.json();
    if (await handleCommand(update, env)) return new Response('ok', {status: 200});
    const file = getFile(update);
    if (!file) return new Response('ignored', {status: 200});

    const chatId = update.message.chat.id;
    const messageId = update.message.message_id;

    // Telegram's normal Bot API file download is limited; don't falsely promise 100 MB here.
    if (file.file_size > 20 * 1024 * 1024) {
      await telegram('sendMessage', env, {
        chat_id: chatId,
        text: '❌ This deployment supports Telegram cloud downloads up to 20 MB. For larger files we will add a local Bot API server later.'
      });
      return new Response('too large', {status: 200});
    }

    try {
      const rows = await supabase('ingest_queue', env, 'POST', {
        telegram_update_id: update.update_id,
        chat_id: chatId,
        message_id: messageId,
        file_id: file.file_id,
        file_name: file.file_name,
        mime_type: file.mime_type,
        file_size: file.file_size
      });

      if (rows && rows.length) {
        await telegram('sendMessage', env, {
          chat_id: chatId,
          text: `📥 Received: ${file.file_name}\n⏳ Added to processing queue.\n\nI will process files one-by-one and save MCQs to Supabase.`
        });
      }
    } catch (e) {
      console.log(e.stack || e);
      await telegram('sendMessage', env, {
        chat_id: chatId,
        text: '⚠️ File received, but queue storage failed. Please try again later.'
      });
    }
    return new Response('ok', {status: 200});
  }
};
