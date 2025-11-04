import base64
import os
import json
import sys
import time
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
MAX_MESSAGES = 500

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