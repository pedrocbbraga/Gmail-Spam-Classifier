#!/usr/bin/env bash
# Fetch the Apache SpamAssassin public corpus used to build the spam-token library.
#
# The corpus is third-party data and is not stored in this repository.
# Run this once before create_spamtoken_dataset.py.
#
#   ./scripts/fetch_data.sh
#
# Produces:
#   data/raw/spam/     ~500 spam messages
#   data/raw/spam_2/   ~1400 spam messages

set -euo pipefail

BASE="https://spamassassin.apache.org/old/publiccorpus"
RAW="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/data/raw"

mkdir -p "$RAW"
cd "$RAW"

for archive in 20021010_spam.tar.bz2 20030228_spam_2.tar.bz2; do
  if [ -f "$archive" ]; then
    echo "already downloaded: $archive"
  else
    echo "downloading $archive ..."
    curl -fSL --retry 3 -O "$BASE/$archive"
  fi
done

echo "extracting ..."
for archive in 20021010_spam.tar.bz2 20030228_spam_2.tar.bz2; do
  tar -xjf "$archive"
done

rm -f spam/cmds spam_2/cmds 2>/dev/null || true

echo
echo "done:"
echo "  spam/    $(find spam   -type f 2>/dev/null | wc -l | tr -d ' ') messages"
echo "  spam_2/  $(find spam_2 -type f 2>/dev/null | wc -l | tr -d ' ') messages"
echo
echo "Corpus: Apache SpamAssassin public mail corpus"
echo "        https://spamassassin.apache.org/old/publiccorpus/readme.html"
