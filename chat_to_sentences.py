"""Convert a claude.ai data export (conversations.json) into a sentences CSV
that main.py can classify.

Usage:
    python chat_to_sentences.py conversations.json chat_sentences.csv
    SENTENCES_CSV=chat_sentences.csv python main.py

Only the user's own messages are exported; assistant replies are skipped.
Labels are left as 'Unknown' because there is no ground truth for chats.
"""
import csv
import json
import re
import sys

MIN_WORDS = 4
MAX_CHARS = 400


def user_messages(conversations):
    for convo in conversations:
        for msg in convo.get("chat_messages", []):
            if msg.get("sender") == "human":
                yield convo.get("name", ""), msg.get("text", "")


def split_sentences(text):
    text = re.sub(r"```.*?```", " ", text, flags=re.S)  # drop code blocks
    for part in re.split(r"(?<=[.!?])\s+|\n+", text):
        part = part.strip()
        if len(part.split()) >= MIN_WORDS and len(part) <= MAX_CHARS:
            yield part


def main(src, dst):
    with open(src, encoding="utf-8") as f:
        conversations = json.load(f)
    rows = 0
    with open(dst, "w", newline="", encoding="utf-8") as out:
        writer = csv.writer(out)
        writer.writerow(["sentence", "expected_role", "expected_tone", "belbin_category", "conversation"])
        for title, text in user_messages(conversations):
            for sentence in split_sentences(text):
                writer.writerow([sentence, "Unknown", "neutral", "unknown", title])
                rows += 1
    print(f"Wrote {rows} sentences to {dst}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(*sys.argv[1:])
