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
MAX_MESSAGES = None

# Get OAuth credentials - boilerplate from Google
# https://developers.google.com/workspace/gmail/api/quickstart/python?utm_source=chatgpt.com
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

