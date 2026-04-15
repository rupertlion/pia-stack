import os
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import io
from google.auth.transport.requests import Request
import requests
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
import PyPDF2
import uuid

# If modifying these SCOPES, delete the file token.json.
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

def authenticate():
    """Shows basic usage of the Drive v3 API.
    Prints the names and ids of up to 10 files.
    """
    creds = None
    # The file token.json stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first
    # time.
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open('token.json', 'w') as token:
            token.write(creds.to_json())

    return creds

def list_files(service):
    """List all files in Google Drive."""
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

    if not all_items:
        print('No files found.')
    else:
        print('Files:')
        for item in all_items:
            print(u'{0} ({1})'.format(item['name'], item['id']))

    return all_items

def download_file(service, file_id):
    """Download a file from Google Drive."""
    request = service.files().get_media(fileId=file_id)
    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while done is False:
        status, done = downloader.next_chunk()
        print("Download %d%%." % int(status.progress() * 100))

    return fh.getvalue()

def extract_text_from_pdf(content, file):
    """Extract text from a PDF file."""
    try:
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(content))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        print(f"Skipping {file['name']}: Error reading PDF structure.")
        return ""

def send_to_embedding_server(text_chunk):
    """Send content to the local embedding server and get vectors."""
    response = requests.post('http://localhost:8081/embed', json={'inputs': [text_chunk]})
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Failed to get embeddings: {response.text}")

def save_to_qdrant(vectors, file_id):
    """Save vectors to the 'documents' collection in Qdrant."""
    client = QdrantClient("localhost", port=6333)
    points = [PointStruct(id=str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{file_id}_{i}")), vector=v) for i, v in enumerate(vectors)]
    if points:
        client.upsert(collection_name="documents", points=points)

def main():
    creds = authenticate()
    service = build('drive', 'v3', credentials=creds)

    files = list_files(service)
    for file in files:
        if file['mimeType'] == 'application/pdf':
            print(f"Ingesting: {file['name']}")
            content = download_file(service, file['id'])
            text = extract_text_from_pdf(content, file)
        elif file['mimeType'] == 'application/vnd.google-apps.document':
            print(f"Ingesting: {file['name']}")
            request = service.files().export(fileId=file['id'], mimeType='text/plain')
            response = request.execute()
            text = response.decode('utf-8')
        elif file['mimeType'] == 'text/plain':
            print(f"Ingesting: {file['name']}")
            content = download_file(service, file['id'])
            text = content.decode('utf-8')
        else:
            continue

        if not text.strip():
            print(f"Skipping {file['name']}: No text content found (possibly a scan).")
            continue

        # Chunk the text into 1000 characters
        chunks = [text[i:i+1000] for i in range(0, len(text), 1000)]
        vectors = []
        for chunk in chunks:
            vectors.extend(send_to_embedding_server(chunk))

        save_to_qdrant(vectors, file['id'])
        print(f"File {file['name']} processed and saved successfully.")

if __name__ == '__main__':
    main()
