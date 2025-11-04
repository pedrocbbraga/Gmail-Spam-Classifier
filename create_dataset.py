import base64
import os
import json
import sys
import time
import tqdm
from typing import List, Dict, Any, Optional
from pathlib import Path
from bs4 import BeautifulSoup

from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from google.auth.transport.requests import Request

# Steps:
# Load config
# Define input query
# get OAuth credentials
# Build GmailAPI client
# List message IDs
# For each ID, get: tokens, URLS, opened status, date

# Load config
SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]
CLIENT_SECRET = "client_secret.json"
TOKEN_FILE = "token.json"
OUT_JSONL = "gmail_dataset.jsonl"

# Input
GMAIL_QUERY  = 'in:inbox newer_than:1y'
MAX_MESSAGES = 50

# Get OAuth credentials - boilerplate from Google
# https://developers.google.com/workspace/gmail/api/quickstart/python
def get_creds() -> Credentials:
    creds = None
    if os.path.exists(TOKEN_FILE):
        creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not os.path.exists(CLIENT_SECRET):
                print("Missing client_secret.json", file=sys.stderr)
                sys.exit(1)
            flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRET, SCOPES)
            creds = flow.run_local_server(port=0, prompt="consent")
        with open(TOKEN_FILE, "w") as f:
            f.write(creds.to_json())
    return creds

# Build Gmail API client - more boilerplate
def build_client(creds: Credentials):
    return build("gmail", "v1", credentials=creds)

# print("check creds")
# creds = get_creds()

# print("build client")
# service = build_client(creds)

def decode_base64url(data: str) -> bytes:
    decoded_bytes = base64.urlsafe_b64decode(data + "===")
    return decoded_bytes

def list_message_ids(service, q: str = "", max_messages: Optional[int] = None) -> List[str]:
    ids = []
    next_page_token = None
    request_args = {"userId": "me", "q": q}

    while True:
        print("here")   

        if next_page_token: # if page token exists, add to request
            request_args["pageToken"] = next_page_token

        results = ( # retrieve 100 ids at a time
                service.users().messages().list(**request_args).execute()
        )

        messages = results.get("messages", [])
        if not messages:
            print("no messages found")
            break

        ids.extend([message["id"] for message in messages]) # collect ids

        if max_messages and len(ids) >= max_messages:
            return ids[:max_messages] # return only up to max_messages if specified

        next_page_token = results.get("nextPageToken", [])

        if not next_page_token:
            break

    if max_messages:
        return ids[:max_messages] # return only up to max_messages if specified
    
    return ids


ids = list_message_ids(build_client(get_creds()), GMAIL_QUERY, MAX_MESSAGES)

# print(ids)
print(f"Found {len(ids)} messages")

def get_message_data(service, msg_id: str) -> Dict[str, Any]:
    message = service.users().messages().get(userId="me", id=msg_id, format="full").execute()
    payload = message.get("payload", {})
    headers = payload.get("headers", [])
    parts = payload.get("parts", [])
    
    # Extract relevant headers
    subject = next((h["value"] for h in headers if h["name"] == "Subject"), "")
    from_email = next((h["value"] for h in headers if h["name"] == "From"), "")
    date = next((h["value"] for h in headers if h["name"] == "Date"), "")
    
    # Extract body content
    body = ""
    for part in parts:
        if part.get("mimeType") == "text/html":
            body_data = part.get("body", {}).get("data", "")
            body = decode_base64url(body_data).decode("utf-8")
            break
        elif part.get("mimeType") == "text/plain":
            body_data = part.get("body", {}).get("data", "")
            body = decode_base64url(body_data).decode("utf-8")
    
    # Parse HTML to extract URLs
    urls = []
    if body:
        soup = BeautifulSoup(body, "html.parser")
        for a in soup.find_all("a", href=True):
            urls.append(a["href"])
    
    return {
        "id": msg_id,
        "subject": subject,
        "from": from_email,
        "date": date,
        "urls": urls,
        "body": body
    }

def create_dataset(service, ids: List[str], out_file: str):
    with open(out_file, "w") as f:
        for i, msg_id in enumerate(ids):
            print(f"Processing message {i+1}/{len(ids)}: {msg_id}")
            msg_data = get_message_data(service, msg_id)
            f.write(json.dumps(msg_data) + "\n")
            time.sleep(0.1)  # To respect API rate limits

create_dataset(build_client(get_creds()), ids, OUT_JSONL)