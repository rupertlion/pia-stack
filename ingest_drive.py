import docx
import pandas as pd
import os
import io
import uuid
import json
import requests
from datetime import datetime
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from google.auth.transport.requests import Request
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
import PyPDF2

# --- CONFIGURATION ---
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']
COLLECTION_NAME = "google_drive"
EMBEDDING_SERVER_URL = "http://localhost:8081/embed"
# Change this to 1024 or 1536 if your local embedding model requires it
VECTOR_SIZE = 768 

# --- INITIALIZE QDRANT ---
qdrant_client = QdrantClient(host="localhost", port=6333)

# Build the vault room if it doesn't exist
if not qdrant_client.collection_exists(collection_name=COLLECTION_NAME):
    print(f"Collection '{COLLECTION_NAME}' not found. Creating it now...")
    qdrant_client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    )
    print(f"Successfully created collection: {COLLECTION_NAME}")

# --- HELPER FUNCTIONS ---

def authenticate():
    creds = None
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        with open('token.json', 'w') as token:
            token.write(creds.to_json())
    return creds

def list_files(service):
    query = "mimeType != 'application/vnd.google-apps.folder' and trashed = false"
    page_token = None
    all_items = []
    while True:
        results = service.files().list(
            q=query,
            fields="nextPageToken, files(id, name, mimeType)",
            pageToken=page_token).execute()
        items = results.get('files', [])
        all_items.extend(items)
        page_token = results.get('nextPageToken')
        if not page_token:
            break
    return all_items

def download_file(service, file_id):
    request = service.files().get_media(fileId=file_id)
    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while done is False:
        status, done = downloader.next_chunk()
    return fh.getvalue()

def extract_text_from_pdf(content, file_name):
    try:
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(content))
        text = ""
        for page in pdf_reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted
        return text
    except Exception as e:
        print(f"Skipping {file_name}: Could not parse PDF structure.")
        return ""

def extract_text_from_docx(content, file_name):
    """Extract text from a Microsoft Word document."""
    try:
        doc = docx.Document(io.BytesIO(content))
        text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
        return text
    except Exception as e:
        print(f"Skipping {file_name}: Could not parse DOCX. Error: {e}")
        return ""

def extract_text_from_xlsx(content, file_name):
    """Extract text from an Excel spreadsheet and format it for the AI."""
    try:
        # Force the openpyxl engine so it doesn't crash on complex formatting
        excel_data = pd.read_excel(io.BytesIO(content), sheet_name=None, engine='openpyxl')
        text = ""
        for sheet_name, df in excel_data.items():
            text += f"--- Sheet: {sheet_name} ---\n"
            text += df.to_markdown(index=False) + "\n\n"
        return text
    except Exception as e:
        print(f"Skipping {file_name}: Could not parse Excel file. Error: {e}")
        return ""

def send_to_embedding_server(text_chunk):
    response = requests.post(EMBEDDING_SERVER_URL, json={'inputs': [text_chunk]})
    if response.status_code == 200:
        # Assuming server returns a list of vectors; we take the first one
        return response.json()[0]
    else:
        raise Exception(f"Embedding failed: {response.text}")

def save_to_qdrant(vector, metadata, stable_id):
    point = PointStruct(
        id=stable_id, 
        vector=vector, 
        payload=metadata
    )
    qdrant_client.upsert(
        collection_name=COLLECTION_NAME,
        wait=True,
        points=[point]
    )

# --- MAIN EXECUTION ---

def main():
    creds = authenticate()
    service = build('drive', 'v3', credentials=creds)

    PROGRESS_FILE = 'ingested_files.json'
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, 'r') as f:
            ingested_ids = set(json.load(f))
    else:
        ingested_ids = set()

    print("Fetching file list from Google Drive...")
    files = list_files(service)
    
    for file in files:
        if file['id'] in ingested_ids:
            print(f"Skipping {file['name']} (already in vault)")
            continue

        text = ""
        
        # Determine File Type & Extract Text
        try:
            # 1. PDFs
            if file['mimeType'] == 'application/pdf':
                print(f"Processing PDF: {file['name']}")
                content = download_file(service, file['id'])
                text = extract_text_from_pdf(content, file['name'])
                
            # 2. Google Docs
            elif file['mimeType'] == 'application/vnd.google-apps.document':
                print(f"Processing Google Doc: {file['name']}")
                request = service.files().export(fileId=file['id'], mimeType='text/plain')
                text = request.execute().decode('utf-8')

            # 3. Google Sheets (Exported instantly as CSV)
            elif file['mimeType'] == 'application/vnd.google-apps.spreadsheet':
                print(f"Processing Google Sheet: {file['name']}")
                request = service.files().export(fileId=file['id'], mimeType='text/csv')
                text = request.execute().decode('utf-8')
                
            # 4. Microsoft Word (.docx)
            elif file['mimeType'] == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
                print(f"Processing Word Doc: {file['name']}")
                content = download_file(service, file['id'])
                text = extract_text_from_docx(content, file['name'])

            # 5. Microsoft Excel (Catches both .xlsx and older .xls)
            elif file['mimeType'] in [
                'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                'application/vnd.ms-excel'
            ]:
                print(f"Processing Excel Sheet: {file['name']}")
                content = download_file(service, file['id'])
                text = extract_text_from_xlsx(content, file['name'])
                
            # 6. Plain Text / Localizable.strings
            elif file['mimeType'] == 'text/plain':
                print(f"Processing Text File: {file['name']}")
                content = download_file(service, file['id'])
                try:
                    text = content.decode('utf-8')
                except UnicodeDecodeError:
                    try:
                        text = content.decode('utf-16')
                    except UnicodeDecodeError:
                        text = content.decode('utf-8', errors='replace')
            else:
                # If it's an image, zip file, or audio, we skip it
                continue
                
        except Exception as e:
            print(f"Failed to download/decode {file['name']}: {e}")
            continue

        if not text or not text.strip():
            print(f"Skipping {file['name']}: No readable text found.")
            continue

        # Chunk, Embed, and Store each segment separately
        # This provides high-resolution context for the Director
        chunks = [text[i:i+1000] for i in range(0, len(text), 1000)]
        
        for index, chunk in enumerate(chunks):
            try:
                vector = send_to_embedding_server(chunk)
                
                # Generate a deterministic ID based on File ID + Chunk Index
                chunk_key = f"{file['id']}_{index}"
                stable_id = str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_key))
                
                metadata = {
                    "drive_id": file['id'],
                    "name": file['name'],
                    "text": chunk, # Storing the text allows the AI to "read" the results
                    "chunk_index": index,
                    "processed_at": str(datetime.now())
                }
                
                save_to_qdrant(vector, metadata, stable_id)
            except Exception as e:
                print(f"Error processing chunk {index} of {file['name']}: {e}")
                continue

        # Mark as ingested only after all chunks are saved
        ingested_ids.add(file['id'])
        with open(PROGRESS_FILE, 'w') as f:
            json.dump(list(ingested_ids), f)

        print(f"✓ {file['name']} fully operational in the vault.")

    print("\n--- Project LION Sync Complete. ---")

if __name__ == '__main__':
    main()