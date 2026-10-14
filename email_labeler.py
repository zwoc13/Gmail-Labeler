#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "google-api-python-client",
#   "google-auth-httplib2",
#   "google-auth-oauthlib",
#   "ollama",
# ]
# ///
"""
Gmail Email Labeler using Ollama + gemma4 (with tool calling)
Auth: OAuth2 for personal Gmail, headless-friendly (paste redirect URL)
Phase 1: samples inbox → gemma4 calls propose_labels() tool with suggested schema
Phase 2: classifies each email → gemma4 calls classify_email() tool with category + reason
"""

import base64
import json
import re
import sys
from pathlib import Path

import ollama
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build

# --- Config ---
SCOPES = ["https://www.googleapis.com/auth/gmail.modify"]
CREDENTIALS_FILE = Path(__file__).parent / "credentials.json"
TOKEN_FILE = Path(__file__).parent / "token.json"
LABELS_CACHE = Path(__file__).parent / "labels.json"
PROCESSED_IDS_FILE = Path(__file__).parent / "processed_ids.json"
RULES_FILE = Path(__file__).parent / "rules.json"

OLLAMA_MODEL = "gemma4:latest"

SAMPLE_SIZE = 30   # emails sampled to propose labels
BATCH_SIZE = 1000    # emails classified per run
LABEL_PREFIX = "AI"


# ---------------------------------------------------------------------------
# Gmail Auth — OAuth2, headless-friendly
# ---------------------------------------------------------------------------
def get_gmail_service():
    creds = None
    if TOKEN_FILE.exists():
        creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not CREDENTIALS_FILE.exists():
                print(
                    "ERROR: credentials.json not found.\n"
                    "Download it from Google Cloud Console → APIs & Services → Credentials\n"
                    "→ Create Credentials → OAuth 2.0 Client ID → Desktop app."
                )
                sys.exit(1)
            flow = Flow.from_client_secrets_file(
                CREDENTIALS_FILE,
                scopes=SCOPES,
                redirect_uri="urn:ietf:wg:oauth:2.0:oob",
            )
            auth_url, _ = flow.authorization_url(access_type="offline", prompt="consent")
            print("\n" + "=" * 60)
            print("Open this URL in your browser (any device):")
            print(f"\n  {auth_url}\n")
            print("Grant access — Google will show you a code on the page.")
            print("Copy that code and paste it below.")
            print("=" * 60 + "\n")
            code = input("Paste the authorization code: ").strip()
            flow.fetch_token(code=code)
            creds = flow.credentials
        TOKEN_FILE.write_text(creds.to_json())
    return build("gmail", "v1", credentials=creds)


# ---------------------------------------------------------------------------
# Email Helpers
# ---------------------------------------------------------------------------
def extract_text(payload: dict) -> str:
    mime = payload.get("mimeType", "")
    if mime == "text/plain":
        data = payload.get("body", {}).get("data", "")
        return base64.urlsafe_b64decode(data + "==").decode("utf-8", errors="replace") if data else ""
    if mime.startswith("multipart/"):
        for part in payload.get("parts", []):
            text = extract_text(part)
            if text:
                return text
    return ""


def get_header(headers: list, name: str) -> str:
    for h in headers:
        if h["name"].lower() == name.lower():
            return h["value"]
    return ""


def fetch_email_summaries(service, count: int) -> list[dict]:
    result = (
        service.users()
        .messages()
        .list(userId="me", labelIds=["INBOX"], maxResults=count)
        .execute()
    )
    summaries = []
    for ref in result.get("messages", []):
        msg = (
            service.users()
            .messages()
            .get(userId="me", id=ref["id"], format="metadata",
                 metadataHeaders=["From", "Subject"])
            .execute()
        )
        headers = msg["payload"].get("headers", [])
        summaries.append({
            "sender": get_header(headers, "From"),
            "subject": get_header(headers, "Subject"),
            "snippet": msg.get("snippet", ""),
        })
    return summaries


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

# Tool the model calls in Phase 1 to submit its label proposals
PROPOSE_LABELS_TOOL = {
    "type": "function",
    "function": {
        "name": "propose_labels",
        "description": (
            "Submit a set of email label categories that best organise this inbox. "
            "Call this once after analysing the provided email sample."
        ),
        "parameters": {
            "type": "object",
            "required": ["labels"],
            "properties": {
                "labels": {
                    "type": "array",
                    "minItems": 4,
                    "maxItems": 8,
                    "description": "Proposed label categories",
                    "items": {
                        "type": "object",
                        "required": ["key", "name", "description"],
                        "properties": {
                            "key": {
                                "type": "string",
                                "description": "UPPER_SNAKE_CASE identifier, e.g. WORK_EMAIL",
                            },
                            "name": {
                                "type": "string",
                                "description": "Short Gmail label name, e.g. Work",
                            },
                            "description": {
                                "type": "string",
                                "description": "One sentence: what kind of emails belong here",
                            },
                        },
                    },
                }
            },
        },
    },
}


def build_classify_tool(labels: list[dict]) -> dict:
    """Return a tool definition whose enum is derived from the proposed labels."""
    return {
        "type": "function",
        "function": {
            "name": "classify_email",
            "description": "Classify an email into exactly one label category.",
            "parameters": {
                "type": "object",
                "required": ["category", "reason"],
                "properties": {
                    "category": {
                        "type": "string",
                        "enum": [lbl["key"] for lbl in labels],
                        "description": "The label key that best fits this email",
                    },
                    "reason": {
                        "type": "string",
                        "description": "One short sentence explaining the classification",
                    },
                },
            },
        },
    }


# ---------------------------------------------------------------------------
# Phase 1 — Label proposal via tool call
# ---------------------------------------------------------------------------
def propose_labels(summaries: list[dict], existing_labels: list[str] | None = None) -> list[dict]:
    lines = [
        f"- From: {s['sender'][:60]} | Subject: {s['subject'][:80]}"
        for s in summaries
    ]

    existing_section = ""
    if existing_labels:
        existing_section = (
            "\n\nThe following labels already exist in Gmail — reuse them where appropriate "
            "and avoid creating redundant ones:\n"
            + "\n".join(f"  - {name}" for name in existing_labels)
            + "\n"
        )

    user_msg = (
        "Analyse the following email sample from my inbox and call propose_labels() "
        "with a set of 4–8 label categories that would best organise it. "
        "Always include a catch-all category last."
        + existing_section
        + "\n\nEmail sample:\n"
        + "\n".join(lines)
    )

    print("Asking gemma4 to analyse your inbox and propose labels...")
    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[{"role": "user", "content": user_msg}],
        tools=[PROPOSE_LABELS_TOOL],
    )

    for call in response.message.tool_calls or []:
        if call.function.name == "propose_labels":
            args = call.function.arguments
            # ollama SDK may return a dict or a JSON string
            if isinstance(args, str):
                args = json.loads(args)
            return args["labels"]

    print("ERROR: Model did not call propose_labels().")
    print("Raw response:", response.message.content)
    sys.exit(1)


def confirm_labels(labels: list[dict]) -> list[dict] | None:
    while True:
        print("\ngemma4 suggests these labels for your inbox:\n")
        for i, lbl in enumerate(labels, 1):
            print(f"  {i}. [{lbl['key']}]  {LABEL_PREFIX}/{lbl['name']}")
            print(f"       {lbl['description']}")
        print()
        choice = input("Accept? [y = yes / r = re-run proposal / q = quit]: ").strip().lower()
        if choice == "y":
            return labels
        if choice == "q":
            sys.exit(0)
        if choice == "r":
            return None


# ---------------------------------------------------------------------------
# Gmail Label Management
# ---------------------------------------------------------------------------
def _create_or_get_label(service, name: str, existing_map: dict) -> str:
    """Create a label by name, returning its ID. If it already exists (409), return existing ID."""
    if name in existing_map:
        return existing_map[name]
    from googleapiclient.errors import HttpError as _HttpError
    try:
        created = service.users().labels().create(
            userId="me",
            body={"name": name, "labelListVisibility": "labelShow",
                  "messageListVisibility": "show"},
        ).execute()
        return created["id"]
    except _HttpError as e:
        if e.status_code == 409:
            # Label exists but wasn't in the initial list (race or casing issue) — re-fetch
            fresh = service.users().labels().list(userId="me").execute().get("labels", [])
            for lbl in fresh:
                if lbl["name"].lower() == name.lower():
                    return lbl["id"]
        raise


def get_or_create_gmail_labels(service, labels: list[dict]) -> dict[str, str]:
    existing = service.users().labels().list(userId="me").execute().get("labels", [])
    existing_map = {lbl["name"]: lbl["id"] for lbl in existing}

    parent_id = _create_or_get_label(service, LABEL_PREFIX, existing_map)
    existing_map[LABEL_PREFIX] = parent_id

    label_ids = {}
    for lbl in labels:
        full_name = f"{LABEL_PREFIX}/{lbl['name']}"
        label_ids[lbl["key"]] = _create_or_get_label(service, full_name, existing_map)
        print(f"  Ready: {full_name}")

    return label_ids


# ---------------------------------------------------------------------------
# Processed IDs — skip emails we've already labelled in a previous run
# ---------------------------------------------------------------------------
def load_processed_ids() -> set[str]:
    if PROCESSED_IDS_FILE.exists():
        return set(json.loads(PROCESSED_IDS_FILE.read_text()))
    return set()


def save_processed_ids(ids: set[str]) -> None:
    PROCESSED_IDS_FILE.write_text(json.dumps(sorted(ids)))


# ---------------------------------------------------------------------------
# Rules — sender-based overrides that bypass AI classification
# Format of rules.json:
# [
#   {"sender_contains": "workgrades.com", "label": "AI/WorkGrades"},
#   {"sender_contains": "github.com",     "label": "AI/Dev"}
# ]
# ---------------------------------------------------------------------------
def load_rules() -> list[dict]:
    if RULES_FILE.exists():
        return json.loads(RULES_FILE.read_text())
    return []


def match_rule(sender: str, rules: list[dict]) -> str | None:
    """Return the label name from the first matching rule, or None."""
    sender_lower = sender.lower()
    for rule in rules:
        if rule.get("sender_contains", "").lower() in sender_lower:
            return rule["label"]
    return None


# ---------------------------------------------------------------------------
# Phase 2 — Classification via tool call
# ---------------------------------------------------------------------------
def classify_email(subject: str, sender: str, body_snippet: str,
                   classify_tool: dict, labels: list[dict]) -> tuple[str, str]:
    """Returns (category_key, reason). Falls back to last label on failure."""
    fallback_key = labels[-1]["key"]

    content = (
        f"From: {sender}\nSubject: {subject}\n\n"
        f"Body preview:\n{body_snippet[:500]}"
    )
    try:
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[{"role": "user", "content": content}],
            tools=[classify_tool],
        )
        for call in response.message.tool_calls or []:
            if call.function.name == "classify_email":
                args = call.function.arguments
                if isinstance(args, str):
                    args = json.loads(args)
                return args["category"], args.get("reason", "")
    except Exception as e:
        print(f"    Ollama error: {e}")

    return fallback_key, "fallback"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=== Gmail Email Labeler ===")
    print(f"Model: {OLLAMA_MODEL}\n")

    try:
        models = ollama.list()
        available = [m.model for m in models.models]
        if not any(OLLAMA_MODEL.split(":")[0] in m for m in available):
            print(f"WARNING: Model '{OLLAMA_MODEL}' not found. Available: {available}")
            print(f"Run: ollama pull {OLLAMA_MODEL}\n")
    except Exception as e:
        print(f"ERROR: Cannot connect to Ollama. Is it running? (ollama serve)\n  {e}")
        sys.exit(1)

    print("Authenticating with Gmail...")
    service = get_gmail_service()
    print("Authenticated.\n")

    # --- Phase 1: determine labels ---
    if LABELS_CACHE.exists():
        cached = json.loads(LABELS_CACHE.read_text())
        print(f"Found cached labels ({len(cached)} labels).")
        reuse = input("Use cached labels? [y / n to re-propose]: ").strip().lower()
        labels = cached if reuse == "y" else None
    else:
        labels = None

    if labels is None:
        print(f"\nSampling {SAMPLE_SIZE} inbox emails for label proposal...")
        summaries = fetch_email_summaries(service, SAMPLE_SIZE)
        print(f"Sampled {len(summaries)} emails.\n")

        # Pass existing AI/* labels so the model doesn't duplicate them
        all_gmail_labels = service.users().labels().list(userId="me").execute().get("labels", [])
        existing_ai_labels = [
            lbl["name"] for lbl in all_gmail_labels
            if lbl["name"].startswith(f"{LABEL_PREFIX}/")
        ]
        if existing_ai_labels:
            print(f"Found {len(existing_ai_labels)} existing {LABEL_PREFIX}/* labels in Gmail.\n")

        while True:
            labels = propose_labels(summaries, existing_ai_labels or None)
            confirmed = confirm_labels(labels)
            if confirmed is not None:
                labels = confirmed
                break

        LABELS_CACHE.write_text(json.dumps(labels, indent=2))
        print(f"Labels saved to {LABELS_CACHE.name}\n")

    # --- Create Gmail labels ---
    print("Setting up Gmail labels...")
    label_ids = get_or_create_gmail_labels(service, labels)
    classify_tool = build_classify_tool(labels)

    # Load rules and ensure their target labels exist in Gmail
    rules = load_rules()
    rule_label_id_cache: dict[str, str] = {}
    if rules:
        existing = service.users().labels().list(userId="me").execute().get("labels", [])
        existing_map = {lbl["name"]: lbl["id"] for lbl in existing}
        for rule in rules:
            lname = rule["label"]
            rule_label_id_cache[lname] = _create_or_get_label(service, lname, existing_map)
        print(f"  Loaded {len(rules)} sender rule(s).")
    print()

    # --- Phase 2: classify inbox ---
    processed_ids = load_processed_ids()
    print(f"Fetching up to {BATCH_SIZE} unprocessed inbox messages...")

    # Paginate until we've collected BATCH_SIZE unseen messages or exhausted the inbox.
    # Gmail returns newest-first, so we walk forward page by page until we find unseen ones.
    messages: list[dict] = []
    page_token = None
    while len(messages) < BATCH_SIZE:
        req = service.users().messages().list(
            userId="me",
            labelIds=["INBOX"],
            maxResults=BATCH_SIZE,
            **( {"pageToken": page_token} if page_token else {}),
        )
        result = req.execute()
        page = result.get("messages", [])
        if not page:
            break

        new = [m for m in page if m["id"] not in processed_ids]
        messages.extend(new[:BATCH_SIZE - len(messages)])

        # If the whole page was already processed, there's nothing newer to find
        if not new:
            break

        page_token = result.get("nextPageToken")
        if not page_token:
            break

    print(f"Found {len(messages)} unprocessed messages.\n")

    stats: dict[str, int] = {lbl["key"]: 0 for lbl in labels}
    stats["rules"] = 0

    ai_label_ids = set(label_ids.values()) | set(rule_label_id_cache.values())

    for i, msg_ref in enumerate(messages, 1):
        msg_id = msg_ref["id"]

        msg = (
            service.users()
            .messages()
            .get(userId="me", id=msg_id, format="full")
            .execute()
        )

        # Already labeled in Gmail (e.g. from a previous run that crashed before saving)
        if set(msg.get("labelIds", [])) & ai_label_ids:
            processed_ids.add(msg_id)
            save_processed_ids(processed_ids)
            continue

        headers = msg["payload"].get("headers", [])
        subject = get_header(headers, "Subject")
        sender = get_header(headers, "From")
        body = extract_text(msg["payload"]) or msg.get("snippet", "")

        print(f"[{i}/{len(messages)}] {sender[:40]!r} | {subject[:50]!r}")

        # Check sender rules before hitting the model
        rule_label = match_rule(sender, rules)
        if rule_label:
            label_id = rule_label_id_cache[rule_label]
            print(f"  → {rule_label}  (rule match)")
            stats["rules"] += 1
        else:
            category, reason = classify_email(subject, sender, body, classify_tool, labels)
            label_id = label_ids[category]
            stats[category] = stats.get(category, 0) + 1
            print(f"  → {category}  ({reason})")

        service.users().messages().modify(
            userId="me",
            id=msg_id,
            body={"addLabelIds": [label_id]},
        ).execute()

        processed_ids.add(msg_id)
        save_processed_ids(processed_ids)

    print("\n=== Done ===")
    for lbl in labels:
        print(f"  {LABEL_PREFIX}/{lbl['name']:<20} {stats.get(lbl['key'], 0)}")
    if stats["rules"]:
        print(f"  Rule matches         : {stats['rules']}")
    print(f"\nFilter by label '{LABEL_PREFIX}/<name>' in Gmail to bulk-select and delete.")


if __name__ == "__main__":
    main()
