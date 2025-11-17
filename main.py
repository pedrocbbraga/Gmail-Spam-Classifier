import os
import sys
import os
import json
from tqdm import tqdm
import pandas as pd
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from google.auth.transport.requests import Request

# Load config
SCOPES = ["https://www.googleapis.com/auth/gmail.modify"]
CLIENT_SECRET = "client_secret.json"
TOKEN_FILE = "token_modify.json"
LABEL_NAME = "TestSpam"

emails_to_move = pd.read_csv("./emails_to_move.csv").to_numpy()
# print(emails_to_move)

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

client = build_client(get_creds())

def get_label(client, label_name: str) -> str:
    labels = client.users().labels().list(userId="me").execute()

    for label in labels["labels"]:
        if label.get("name") == label_name:
            label_id = label.get("id")
            break
    
    return label_id

print(get_label(client, LABEL_NAME))
label_id = get_label(client, LABEL_NAME)

inbox_id = get_label(client, "INBOX")

def move_to_label(client, id: str, label_id: str):
    body = {"addLabelIds": label_id}

    return client.users().messages().modify(userId="me", id=id, body=body).execute()

# Essentially, remove inbox label
def archive(client, id: str, inbox_id: str):
    body = {"removeLabelIds": inbox_id}

    return client.users().messages().modify(userId="me", id=id, body=body).execute()

if __name__ == "__main__":
    for id in tqdm(emails_to_move[2000:]):
        print(id[0])
        # move_to_label(client, id[0], label_id)
        # archive(client, id[0], inbox_id)
        move_to_label(client, id[0], inbox_id)
        archive(client, id[0], label_id)
