"""

create_input_dataset.py: This module creates the input dataset for spam_filter.py

It gets authorization to access Gmail, retrieves email metadata, and processes emails to extract features.

Inputs: Gmail data via Gmail API, spam token library
Output: email dataset in JSONL format

Usage: e.g., from the root Gmail-Spam-Classifier directory, python3 source/create_input_dataset.py --N 500

"""

import base64
import os
import json
import sys
import time
import tqdm
import unicodedata
import re
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from bs4 import BeautifulSoup
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from google.auth.transport.requests import Request
from config import dataset_path
import argparse

# Steps:
# Load config
# Define input query
# get OAuth credentials
# Build GmailAPI client
# List message IDs
# For each ID, get: tokens, URLS, opened status, date

parser = argparse.ArgumentParser()
parser.add_argument("--N", type=int, default=500)
args = parser.parse_args()

N = args.N

# Config
SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]
CLIENT_SECRET = "client_secret.json"
TOKEN_FILE = "token_read.json"
OUT_JSONL = dataset_path(N)

# Input
GMAIL_QUERY  = 'in:inbox newer_than:1y'
MAX_MESSAGES = N

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

# Helper function to strip emails from names
def strip_email(email_str: str) -> str:
    if "<" in email_str and ">" in email_str:
        start = email_str.index("<") + 1
        end = email_str.index(">")
        return email_str[start:end]
    return email_str

# Helper function to clean invisible characters
# https://www.fileformat.info/info/unicode/category/index.htm
def clean_invisible_chars(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    cleaned_characters = []
    
    for character in text:
        category = unicodedata.category(character)

        if category[0] in ("C"): # control characters
            cleaned_characters.append(character)
        elif category.startswith(("Mn", "Me")) or category == "Cf": # non-spacing marks, enclosing marks, format characters
            cleaned_characters.append(" ")
        elif category[0] == "Z": # separator characters
            cleaned_characters.append(" ")
        else:
            cleaned_characters.append(character)
        
    cleaned_body_la_ele = "".join(cleaned_characters)
    cleaned_body_la_ele = re.sub(r"[\u00AD\u200B-\u200F\u202A-\u202E\u2060-\u206F\uFEFF\u034F]+", "", cleaned_body_la_ele)

    return " ".join(cleaned_body_la_ele.split())

def open_spamtoken_library(csv_path: str) -> List[str]:
    data = pd.read_csv(csv_path)
    return data['token'].tolist()

# Helper function to count spammy tokens in body
def count_spammy_tokens(body: str, spam_tokens: List[str]) -> int:
    count = 0
    body_lower = [word.lower() for word in body.split()]
    spam_tokens_lower = [token.lower() for token in spam_tokens]
    for token in spam_tokens_lower:
        count += body_lower.count(token)
    return count

# Get email data
def get_message_data(service, msg_id: str) -> Dict[str, Any]:
    message = service.users().messages().get(userId="me", id=msg_id, format="full").execute()
    payload = message.get("payload", {})
    headers = payload.get("headers", [])
    parts = payload.get("parts", [])
    
    # Extract relevant headers
    subject = next((h["value"] for h in headers if h["name"] == "Subject"), "")
    from_email = next((h["value"] for h in headers if h["name"] == "From"), "")
    from_email = strip_email(from_email)
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
    
    # Parse HTML to extract URLs and clean up body text
    urls = []
    clean_body_la_ele = ""
    if body:
        soup = BeautifulSoup(body, "html.parser")

        for tag in soup(["script", "style", "head", "meta", "title"]):
            tag.decompose()  # remove tags

        for a in soup.find_all("a", href=True):
            urls.append(a["href"])

        clean_body_la_ele = soup.get_text(separator=" ", strip=True)
        clean_body_la_ele = clean_invisible_chars(clean_body_la_ele)

    body = clean_body_la_ele
    spamtoken_count = count_spammy_tokens(body, open_spamtoken_library("spamtoken_library.csv"))
    url_count = len(urls)


    if 'UNREAD' in message.get('labelIds', []):
        opened = False
    else:
        opened = True
    
    return {
        "id": msg_id,
        "subject": subject,
        "from": from_email,
        "date": date,
        "url count": url_count,
        "spam token count": spamtoken_count,
        "opened": opened
    }

def create_dataset(service, ids: List[str], out_file: str):
    with open(out_file, "w") as f:
        for i, msg_id in enumerate(ids):
            print(f"Processing message {i+1}/{len(ids)}: {msg_id}")
            msg_data = get_message_data(service, msg_id)
            json.dump(msg_data, f, ensure_ascii=False)
            f.write("\n")
            time.sleep(0.1)  # to respect API rate limitation

create_dataset(build_client(get_creds()), ids, OUT_JSONL)