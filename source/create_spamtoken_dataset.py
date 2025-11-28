"""

create_spamtoken_dataset.py: This module creates the input dataset for create_spamtoken_library.py

It gets retrieves email metadata from a spam email dataset (pulled from Apache SpamAssassin)
and processes the data to extract email bodies.

Inputs: spam email dataset
Output: email bodies in JSONL format

Usage: e.g., from the root Gmail-Spam-Classifier directory, python3 source/create_spamtoken_dataset.py

"""

import os
import json
import time
import unicodedata
import re
from pathlib import Path
from bs4 import BeautifulSoup
from email import policy
from email.parser import BytesParser
from config import SPAM_RAW, SPAM_RAW_2, DATA_RAW

# File paths
IDS_PATH = SPAM_RAW
# IDS_PATH = SPAM_RAW_2  # if using second spam dataset
OUT_JSONL = DATA_RAW / "spamtoken_dataset.jsonl"

ids = os.listdir(IDS_PATH)

# print(ids)
# print(f"Found {len(ids)} messages")

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

# Get email data from files
def get_file_email_body(path: str) -> str:
    # Read raw email bytes
    with open(path, "rb") as f:
        raw = f.read()

    if raw.startswith(b"From "):
        first_nl = raw.find(b"\n")
        if first_nl != -1:
            raw = raw[first_nl + 1 :]

    # Parse email using stdlib
    msg = BytesParser(policy=policy.default).parsebytes(raw)
    # print(msg)

    text_plain = None
    text_html = None

    if msg.is_multipart():
        # print("here")
        for part in msg.walk():
            content_disposition = (part.get_content_disposition() or "").lower()
            if content_disposition == "attachment":
                continue  # skip attachments

            ctype = part.get_content_type()

            if ctype == "text/html" and text_html is None:
                payload = part.get_payload(decode=True)
                if payload is not None:
                    try:
                        text_html = payload.decode(part.get_content_charset() or "utf-8", errors="replace")
                    except LookupError:
                        text_html = payload.decode("utf-8", errors="replace")

            elif ctype == "text/plain" and text_plain is None:
                payload = part.get_payload(decode=True)
                if payload is not None:
                    try:
                        text_plain = payload.decode(part.get_content_charset() or "utf-8", errors="replace")
                    except LookupError:
                        text_plain = payload.decode("utf-8", errors="replace")
    else:
        # print("here2")
        ctype = msg.get_content_type()
        payload = msg.get_payload(decode=True)
        if payload is not None:
            if ctype == "text/html":
                try:
                    text_html = payload.decode(msg.get_content_charset() or "utf-8", errors="replace")
                except LookupError:
                    text_html = payload.decode("utf-8", errors="replace")
            elif ctype == "text/plain":
                try:
                    text_plain = payload.decode(msg.get_content_charset() or "utf-8", errors="replace")
                except LookupError:
                    text_plain = payload.decode("utf-8", errors="replace")

    # Clean and normalize body
    # Prefer HTML over plain text
    if text_html:
        soup = BeautifulSoup(text_html, "html.parser")
        for tag in soup(["script", "style", "head", "meta", "title"]):
            tag.decompose()
        body = soup.get_text(separator=" ", strip=True)
        body = clean_invisible_chars(body)
        return body

    if text_plain:
        return clean_invisible_chars(text_plain)

    return ""

def create_dataset(ids_path: str, out_file: str):
    with open(out_file, "w") as f:
        i = 1
        for msg_id in os.listdir(ids_path):
            file_path = os.path.join(ids_path, msg_id)
            print(f"Processing message {i+1}/{len(ids)}: {msg_id}")
            msg_data = get_file_email_body(file_path)
            json.dump(msg_data, f, ensure_ascii=False)
            f.write("\n")
            time.sleep(0.1)  # to respect API rate limitation
            i += 1

create_dataset(IDS_PATH, OUT_JSONL)