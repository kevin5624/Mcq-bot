import os, requests, sys
url=os.environ['SUPABASE_URL_1'].rstrip('/')
key=os.environ['SUPABASE_KEY_1']
chat=os.environ['ADMIN_CHAT_ID']
r=requests.post(url+'/rest/v1/ingest_queue', headers={'apikey':key,'Authorization':'Bearer '+key,'Content-Type':'application/json','Prefer':'return=minimal'}, json={'chat_id':int(chat),'job_type':'scrape','target_count':500,'file_id':None,'file_name':None,'mime_type':None,'file_size':0}, timeout=30)
r.raise_for_status()
print('Daily scrape job queued')
