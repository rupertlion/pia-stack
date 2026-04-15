import hashlib
import json
import os
import sys
import base64
from datetime import datetime, timedelta
from pathlib import Path
from email.utils import parsedate_to_datetime

try:
    import httpx
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install",
        "httpx", "qdrant-client", "tqdm",
        "google-auth-oauthlib", "google-api-python-client"])
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build
    import httpx

from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct, PayloadSchemaType, TextIndexParams
from tqdm import tqdm

QDRANT_URL = "http://localhost:6333"
TEI_URL = "http://localhost:8081"
EMBED_DIM = 768
BATCH_SIZE = 8  # <--- CHANGED FROM 32
SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/calendar.readonly",
]
SCRIPT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
CREDS_FILE = SCRIPT_DIR / "google_credentials.json"
TOKEN_FILE = SCRIPT_DIR / "google_token.json"

cli_args = sys.argv[1:]
SKIP_CALENDAR = "--no-calendar" in cli_args
SKIP_EMAIL = "--no-email" in cli_args
DAYS_BACK = 365
if "--days" in cli_args:
    idx = cli_args.index("--days")
    DAYS_BACK = int(cli_args[idx + 1])


def get_google_creds():
    creds = None
    if TOKEN_FILE.exists():
        creds = Credentials.from_authorized_user_file(str(TOKEN_FILE), SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not CREDS_FILE.exists():
                print("Missing " + str(CREDS_FILE))
                print("Download OAuth credentials from Google Cloud Console.")
                sys.exit(1)
            flow = InstalledAppFlow.from_client_secrets_file(str(CREDS_FILE), SCOPES)
            creds = flow.run_local_server(port=0)
        with open(TOKEN_FILE, "w") as f:
            f.write(creds.to_json())
    return creds


def setup_collection(name, client):
    if client.collection_exists(name):
        info = client.get_collection(name)
        print("  Collection " + name + " exists (" + str(info.points_count) + " vectors)")
    else:
        print("  Creating collection " + name + "...")
        client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
        )
        client.create_payload_index(name, "sender", field_schema=PayloadSchemaType.KEYWORD)
        client.create_payload_index(name, "folder", field_schema=PayloadSchemaType.KEYWORD)
        client.create_payload_index(
            name, "subject",
            field_schema=TextIndexParams(type="text", tokenizer="word", min_token_len=2, max_token_len=20)
        )


def get_gmail_messages(service, days_back=365):
    after_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y/%m/%d")
    query = "after:" + after_date
    emails = []
    page_token = None
    total = 0
    print("  Fetching Gmail messages since " + after_date + "...")
    while True:
        results = service.users().messages().list(
            userId="me", q=query, maxResults=100, pageToken=page_token
        ).execute()
        messages = results.get("messages", [])
        for msg_ref in messages:
            try:
                msg = service.users().messages().get(
                    userId="me", id=msg_ref["id"], format="full"
                ).execute()
                headers = {}
                for h in msg["payload"]["headers"]:
                    headers[h["name"].lower()] = h["value"]
                subject = headers.get("subject", "No Subject")
                sender = headers.get("from", "unknown")
                to = headers.get("to", "")
                date = headers.get("date", "")
                body = ""
                payload = msg["payload"]
                if "parts" in payload:
                    for part in payload["parts"]:
                        if part["mimeType"] == "text/plain":
                            bdata = part.get("body", {}).get("data", "")
                            if bdata:
                                body = base64.urlsafe_b64decode(bdata).decode("utf-8", errors="ignore")
                                break
                elif "body" in payload:
                    bdata = payload["body"].get("data", "")
                    if bdata:
                        body = base64.urlsafe_b64decode(bdata).decode("utf-8", errors="ignore")
                if len(body.strip()) < 20:
                    continue
                try:
                    dt = parsedate_to_datetime(date)
                    date_iso = dt.strftime("%Y-%m-%dT%H:%M:%S")
                except Exception:
                    date_iso = date
                labels = msg.get("labelIds", [])
                folder = "Gmail/" + "/".join(labels[:2]) if labels else "Gmail"
                emails.append({
                    "subject": subject, "sender": sender, "to": to,
                    "date": date_iso, "body": body, "folder": folder,
                })
                total += 1
                if total % 100 == 0:
                    print("    ..." + str(total) + " Gmail messages")
            except Exception:
                continue
        page_token = results.get("nextPageToken")
        if not page_token:
            break
    print("  " + str(total) + " Gmail messages fetched")
    return emails


def get_gcal_events(service, days_back=365):
    time_min = (datetime.utcnow() - timedelta(days=days_back)).isoformat() + "Z"
    time_max = (datetime.utcnow() + timedelta(days=90)).isoformat() + "Z"
    events = []
    page_token = None
    print("  Fetching Google Calendar events...")
    while True:
        results = service.events().list(
            calendarId="primary", timeMin=time_min, timeMax=time_max,
            maxResults=250, singleEvents=True,
            orderBy="startTime", pageToken=page_token
        ).execute()
        for item in results.get("items", []):
            start_info = item.get("start", {})
            end_info = item.get("end", {})
            start = start_info.get("dateTime", start_info.get("date", ""))
            end = end_info.get("dateTime", end_info.get("date", ""))
            att_list = item.get("attendees", [])
            attendees = ", ".join([a.get("email", "") for a in att_list])
            org = item.get("organizer", {})
            events.append({
                "subject": item.get("summary", "No Title"),
                "start": start, "end": end,
                "location": item.get("location", ""),
                "body": item.get("description", ""),
                "organizer": org.get("email", ""),
                "attendees": attendees,
            })
        page_token = results.get("nextPageToken")
        if not page_token:
            break
    print("  " + str(len(events)) + " Google Calendar events fetched")
    return events


def chunk_email(email, max_tokens=400):
    words = email["body"].split()
    chunks = []
    step = max_tokens - 50
    for i in range(0, len(words), step):
        chunk_text = " ".join(words[i:i + max_tokens])
        subj = email["subject"]
        sndr = email["sender"]
        fldr = email["folder"]
        enriched = "Subject: " + subj + "\nFrom: " + sndr + "\nFolder: " + fldr + "\n\n" + chunk_text
        raw_id = "gmail:" + sndr + ":" + email["date"] + ":" + str(i)
        chunk_id = hashlib.md5(raw_id.encode()).hexdigest()
        chunks.append({
            "id": chunk_id, "text": enriched,
            "metadata": {
                "subject": subj, "sender": sndr,
                "to": email["to"], "date": email["date"],
                "folder": fldr, "type": "gmail",
                "chunk_index": i // step,
            }
        })
    return chunks


def chunk_calendar_event(event):
    text_parts = [
        "Calendar Event: " + event["subject"],
        "Start: " + event["start"],
        "End: " + event["end"],
    ]
    if event["location"]:
        text_parts.append("Location: " + event["location"])
    if event["organizer"]:
        text_parts.append("Organizer: " + event["organizer"])
    if event["attendees"]:
        text_parts.append("Attendees: " + event["attendees"])
    if event["body"]:
        text_parts.append(" ".join(event["body"].split()[:300]))
    raw_id = "gcal:" + event["subject"] + ":" + event["start"]
    chunk_id = hashlib.md5(raw_id.encode()).hexdigest()
    return [{
        "id": chunk_id,
        "text": "\n".join(text_parts),
        "metadata": {
            "subject": event["subject"],
            "start": event["start"],
            "end": event["end"],
            "location": event["location"],
            "organizer": event["organizer"],
            "type": "gcal_event",
            "folder": "Google Calendar",
        }
    }]


def embed_batch(texts):
    resp = httpx.post(
        TEI_URL + "/embed",
        json={"inputs": texts, "truncate": True},
        timeout=600.0  # <--- CHANGED FROM 120.0
    )
    resp.raise_for_status()
    return resp.json()


def ingest_chunks(client, collection, chunks):
    for i in tqdm(range(0, len(chunks), BATCH_SIZE), desc="Embedding -> " + collection):
        batch = chunks[i:i + BATCH_SIZE]
        texts = [c["text"] for c in batch]
        try:
            embeddings = embed_batch(texts)
        except Exception as e:
            print("  Embedding failed: " + str(e))
            continue
        points = [
            PointStruct(
                id=c["id"], vector=emb,
                payload=dict(list(c["metadata"].items()) + [("content", c["text"])])
            )
            for c, emb in zip(batch, embeddings)
        ]
        client.upsert(collection, points)


def main():
    print("=" * 60)
    print("  Gmail + Google Calendar -> Qdrant")
    print("=" * 60)

    creds = get_google_creds()
    qclient = QdrantClient(url=QDRANT_URL)

    if not SKIP_EMAIL:
        setup_collection("emails", qclient)
        gmail = build("gmail", "v1", credentials=creds)
        emails = get_gmail_messages(gmail, days_back=DAYS_BACK)
        all_chunks = []
        for email in tqdm(emails, desc="Chunking Gmail"):
            all_chunks.extend(chunk_email(email))
        print("  Gmail chunks: " + str(len(all_chunks)))
        if all_chunks:
            ingest_chunks(qclient, "emails", all_chunks)

    if not SKIP_CALENDAR:
        setup_collection("calendar", qclient)
        gcal = build("calendar", "v3", credentials=creds)
        events = get_gcal_events(gcal, days_back=DAYS_BACK)
        all_chunks = []
        for event in events:
            all_chunks.extend(chunk_calendar_event(event))
        print("  GCal chunks: " + str(len(all_chunks)))
        if all_chunks:
            ingest_chunks(qclient, "calendar", all_chunks)

    print("")
    print("=" * 60)
    for col in ["emails", "calendar"]:
        try:
            count = qclient.count(col).count
            print("  " + col + ": " + str(count) + " vectors")
        except Exception:
            pass
    print("=" * 60)


if __name__ == "__main__":
    main()