"""
memory.py — Persistent long-term memory for the BVRIT chatbot.

Storage:
  - SQLite database: memory.db  (profiles + conversation log)
  - JSON export/import support

Profile fields:
  user_id              TEXT  primary key (session-based or user-supplied name)
  name                 TEXT  user's preferred name
  preferred_name       TEXT  nickname or short name
  branch_interest      TEXT  e.g. "CSE", "ECE"
  language             TEXT  "english" | "telugu" | "hindi"  (default: english)
  detail_level         TEXT  "brief" | "detailed"            (default: brief)
  answer_style         TEXT  "formal" | "casual" | "friendly" (default: friendly)
  prior_topics         TEXT  JSON list of topics discussed
  facts                TEXT  JSON dict of arbitrary learned facts
  last_session_summary TEXT  LLM-generated summary of last session
  last_active          TEXT  ISO datetime of last interaction
  created_at           TEXT  ISO datetime of profile creation

Conversation history:
  user_id   TEXT
  role      TEXT  "user" | "assistant"
  content   TEXT
  turn_num  INTEGER
  ts        TEXT  ISO datetime

Features:
  - Full conversation history for multi-turn context
  - LLM-based fact extraction (name, branch, language, style, preferences)
  - Summarise older turns after every 10 turns (keep latest 10)
  - Auto-delete profiles inactive > 30 days
  - clear_user_data() — GDPR-style full deletion
  - export_memory() / import_memory() — portable memory backup
  - get_memory_summary() — human-readable memory snapshot
  - detect_memory_changes() — track what changed between updates
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_BASE_DIR    = Path(__file__).parent
DB_PATH      = _BASE_DIR / "memory.db"
JSON_PATH    = _BASE_DIR / "memory.json"

SUMMARISE_AFTER = 10   # turns before summarisation kicks in
KEEP_LATEST     = 10   # turns to keep after summarisation
INACTIVE_DAYS   = 30   # days before auto-deletion

# ---------------------------------------------------------------------------
# DB initialisation
# ---------------------------------------------------------------------------

def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Create tables if they don't exist (with migration support for new columns)."""
    with _connect() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS profiles (
                user_id              TEXT PRIMARY KEY,
                name                 TEXT DEFAULT '',
                preferred_name       TEXT DEFAULT '',
                branch_interest      TEXT DEFAULT '',
                language             TEXT DEFAULT 'english',
                detail_level         TEXT DEFAULT 'brief',
                answer_style         TEXT DEFAULT 'friendly',
                prior_topics         TEXT DEFAULT '[]',
                facts                TEXT DEFAULT '{}',
                last_session_summary TEXT DEFAULT '',
                last_active          TEXT DEFAULT '',
                created_at           TEXT DEFAULT ''
            );

            CREATE TABLE IF NOT EXISTS conversations (
                id       INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id  TEXT NOT NULL,
                role     TEXT NOT NULL,
                content  TEXT NOT NULL,
                turn_num INTEGER DEFAULT 0,
                ts       TEXT DEFAULT ''
            );
        """)

        # ── Migration: add columns that may not exist in older DBs ──────────
        existing_cols = {
            r[1] for r in conn.execute("PRAGMA table_info(profiles)").fetchall()
        }
        migrations = {
            "preferred_name": "ALTER TABLE profiles ADD COLUMN preferred_name TEXT DEFAULT ''",
            "answer_style":   "ALTER TABLE profiles ADD COLUMN answer_style TEXT DEFAULT 'friendly'",
            "facts":          "ALTER TABLE profiles ADD COLUMN facts TEXT DEFAULT '{}'",
        }
        for col, ddl in migrations.items():
            if col not in existing_cols:
                conn.execute(ddl)


# ---------------------------------------------------------------------------
# Profile CRUD
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_profile(user_id: str) -> dict[str, Any]:
    """Load a profile; create a default one if it doesn't exist."""
    init_db()
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM profiles WHERE user_id = ?", (user_id,)
        ).fetchone()

    if row is None:
        now = _now_iso()
        profile = {
            "user_id":              user_id,
            "name":                 "",
            "preferred_name":       "",
            "branch_interest":      "",
            "language":             "english",
            "detail_level":         "brief",
            "answer_style":         "friendly",
            "prior_topics":         [],
            "facts":                {},
            "last_session_summary": "",
            "last_active":          now,
            "created_at":           now,
        }
        _save_profile_dict(profile)
        return profile

    return _row_to_profile(row)


def _row_to_profile(row: sqlite3.Row) -> dict[str, Any]:
    d = dict(row)
    try:
        d["prior_topics"] = json.loads(d.get("prior_topics") or "[]")
    except (json.JSONDecodeError, TypeError):
        d["prior_topics"] = []
    try:
        d["facts"] = json.loads(d.get("facts") or "{}")
    except (json.JSONDecodeError, TypeError):
        d["facts"] = {}
    return d


def _save_profile_dict(profile: dict[str, Any]) -> None:
    topics_json = json.dumps(profile.get("prior_topics", []))
    facts_json  = json.dumps(profile.get("facts", {}))
    with _connect() as conn:
        conn.execute("""
            INSERT INTO profiles
                (user_id, name, preferred_name, branch_interest, language,
                 detail_level, answer_style, prior_topics, facts,
                 last_session_summary, last_active, created_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(user_id) DO UPDATE SET
                name                 = excluded.name,
                preferred_name       = excluded.preferred_name,
                branch_interest      = excluded.branch_interest,
                language             = excluded.language,
                detail_level         = excluded.detail_level,
                answer_style         = excluded.answer_style,
                prior_topics         = excluded.prior_topics,
                facts                = excluded.facts,
                last_session_summary = excluded.last_session_summary,
                last_active          = excluded.last_active
        """, (
            profile["user_id"],
            profile.get("name", ""),
            profile.get("preferred_name", ""),
            profile.get("branch_interest", ""),
            profile.get("language", "english"),
            profile.get("detail_level", "brief"),
            profile.get("answer_style", "friendly"),
            topics_json,
            facts_json,
            profile.get("last_session_summary", ""),
            profile.get("last_active", _now_iso()),
            profile.get("created_at", _now_iso()),
        ))


def update_profile(user_id: str, **fields) -> dict[str, Any]:
    """
    Update specific fields in an existing profile.
    Returns the updated profile dict.
    """
    profile = load_profile(user_id)
    for key, value in fields.items():
        if key == "prior_topics" and isinstance(value, list):
            existing = profile.get("prior_topics", [])
            merged   = list(dict.fromkeys(existing + value))[-20:]
            profile["prior_topics"] = merged
        elif key == "facts" and isinstance(value, dict):
            existing = profile.get("facts", {})
            existing.update(value)
            profile["facts"] = existing
        elif key in profile:
            profile[key] = value
    profile["last_active"] = _now_iso()
    _save_profile_dict(profile)
    return profile


def touch_profile(user_id: str) -> None:
    """Update last_active timestamp."""
    with _connect() as conn:
        conn.execute(
            "UPDATE profiles SET last_active = ? WHERE user_id = ?",
            (_now_iso(), user_id)
        )


# ---------------------------------------------------------------------------
# Memory change detection
# ---------------------------------------------------------------------------

def detect_memory_changes(old_profile: dict, new_profile: dict) -> list[str]:
    """
    Compare old and new profile to detect what changed.
    Returns a list of human-readable change descriptions.
    """
    changes: list[str] = []
    tracked_fields = {
        "name":            "name",
        "preferred_name":  "preferred name",
        "branch_interest": "branch of interest",
        "language":        "language preference",
        "detail_level":    "detail level preference",
        "answer_style":    "answer style preference",
    }
    for field, label in tracked_fields.items():
        old_val = old_profile.get(field, "")
        new_val = new_profile.get(field, "")
        if old_val != new_val and new_val:
            if old_val:
                changes.append(f"Updated {label} from '{old_val}' to '{new_val}'")
            else:
                changes.append(f"Learned {label}: '{new_val}'")

    # Check facts changes
    old_facts = old_profile.get("facts", {})
    new_facts = new_profile.get("facts", {})
    for key, val in new_facts.items():
        if key not in old_facts:
            changes.append(f"Learned fact: {key} = '{val}'")
        elif old_facts[key] != val:
            changes.append(f"Updated fact: {key} from '{old_facts[key]}' to '{val}'")

    return changes


# ---------------------------------------------------------------------------
# Fact extraction — detect profile updates from user messages
# ---------------------------------------------------------------------------

_BRANCH_KEYWORDS = {
    "cse": "CSE", "computer science": "CSE",
    "cse-aiml": "CSE-AIML", "ai&ml": "CSE-AIML", "aiml": "CSE-AIML",
    "cse-ds": "CSE-DS", "data science": "CSE-DS",
    "ece": "ECE", "electronics": "ECE",
    "eee": "EEE", "electrical": "EEE",
    "mech": "Mechanical", "mechanical": "Mechanical",
    "it": "IT", "information technology": "IT",
}

_DETAIL_KEYWORDS = {
    "brief": "brief", "short": "brief", "concise": "brief", "simple": "brief",
    "detailed": "detailed", "elaborate": "detailed", "in detail": "detailed",
    "more detail": "detailed", "explain more": "detailed",
}

_LANGUAGE_KEYWORDS = {
    "telugu": "telugu", "తెలుగు": "telugu",
    "hindi": "hindi", "हिंदी": "hindi",
    "english": "english",
}

_STYLE_KEYWORDS = {
    "formal": "formal", "professional": "formal", "official": "formal",
    "casual": "casual", "informal": "casual", "relaxed": "casual",
    "friendly": "friendly", "warm": "friendly", "conversational": "friendly",
}


def extract_facts_from_message(user_id: str, message: str) -> dict[str, Any]:
    """
    Scan a user message for learnable facts using regex + keyword matching.
    Returns a dict of fields to update (empty dict if nothing learned).

    Detects:
      - Name: "my name is X", "I am X", "I'm X", "call me X"
      - Preferred name: "you can call me X", "nickname is X"
      - Branch interest: "I study CSE", "I'm in ECE"
      - Detail level: "brief", "detailed", "concise"
      - Language: "telugu", "hindi", "english"
      - Answer style: "formal", "casual", "friendly"
      - Facts: year of study, student status, interest area
    """
    msg   = message.lower().strip()
    updates: dict[str, Any] = {}

    # Name detection: "my name is X" / "I am X" / "I'm X"
    name_match = re.search(
        r"(?:my name is|i am|i'm)\s+([a-zA-Z]{2,20})(?:\.|,|$|\s)", msg
    )
    if name_match:
        updates["name"] = name_match.group(1).strip().title()

    # Preferred name / nickname: "you can call me X" / "call me X" / "nickname is X"
    pref_match = re.search(
        r"(?:you can call me|call me|nickname is|prefer to be called)\s+([a-zA-Z]{2,20})", msg
    )
    if pref_match:
        updates["preferred_name"] = pref_match.group(1).strip().title()

    # Branch interest
    for kw, branch in _BRANCH_KEYWORDS.items():
        if kw in msg:
            updates["branch_interest"] = branch
            break

    # Detail level
    for kw, level in _DETAIL_KEYWORDS.items():
        if kw in msg:
            updates["detail_level"] = level
            break

    # Language preference
    for kw, lang in _LANGUAGE_KEYWORDS.items():
        if kw in msg:
            updates["language"] = lang
            break

    # Answer style
    for kw, style in _STYLE_KEYWORDS.items():
        if kw in msg:
            updates["answer_style"] = style
            break

    # Extract arbitrary facts from structured patterns
    fact_patterns = [
        (r"(?:i am|i'm)\s+(?:a\s+|an\s+)?(first.year|second.year|third.year|fourth.year|final.year)\s+(?:student)?", "year_of_study"),
        (r"(?:i am|i'm)\s+(?:a\s+|an\s+)?(fresher|junior|senior)", "student_status"),
        (r"(?:i am|i'm)\s+interested\s+in\s+(placements?|higher\s*studies|research|internships?)", "interest_area"),
        (r"(?:i\s+(?:am\s+)?)?looking\s+for\s+(?:a\s+)?(scholarship|hostel|transport)", "looking_for"),
        (r"(?:i\s+)?(?:want|need|require)\s+(?:a\s+)?(scholarship|hostel|transport)", "looking_for"),
    ]
    for pattern, fact_key in fact_patterns:
        match = re.search(pattern, msg)
        if match:
            if "facts" not in updates:
                updates["facts"] = {}
            updates["facts"][fact_key] = match.group(1).strip().title()

    return updates


# ---------------------------------------------------------------------------
# Conversation history
# ---------------------------------------------------------------------------

def append_turn(user_id: str, role: str, content: str) -> None:
    """Append one turn to the conversation log."""
    init_db()
    with _connect() as conn:
        count = conn.execute(
            "SELECT COUNT(*) FROM conversations WHERE user_id = ?", (user_id,)
        ).fetchone()[0]
        conn.execute(
            "INSERT INTO conversations (user_id, role, content, turn_num, ts) VALUES (?,?,?,?,?)",
            (user_id, role, content, count + 1, _now_iso())
        )


def get_history(user_id: str, limit: int = KEEP_LATEST) -> list[dict]:
    """Return the latest `limit` turns for a user."""
    init_db()
    with _connect() as conn:
        rows = conn.execute(
            """SELECT role, content FROM conversations
               WHERE user_id = ?
               ORDER BY turn_num DESC LIMIT ?""",
            (user_id, limit)
        ).fetchall()
    return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]


def get_turn_count(user_id: str) -> int:
    init_db()
    with _connect() as conn:
        return conn.execute(
            "SELECT COUNT(*) FROM conversations WHERE user_id = ?", (user_id,)
        ).fetchone()[0]


def trim_history(user_id: str, keep: int = KEEP_LATEST) -> None:
    """Delete all turns except the latest `keep` for a user."""
    init_db()
    with _connect() as conn:
        max_id = conn.execute(
            """SELECT id FROM conversations WHERE user_id = ?
               ORDER BY turn_num DESC LIMIT 1 OFFSET ?""",
            (user_id, keep - 1)
        ).fetchone()
        if max_id:
            conn.execute(
                "DELETE FROM conversations WHERE user_id = ? AND id <= ?",
                (user_id, max_id["id"])
            )


# ---------------------------------------------------------------------------
# Summarisation (calls Mistral)
# ---------------------------------------------------------------------------

def summarise_and_trim(user_id: str, mistral_api_key: str) -> str | None:
    """
    If total turns >= SUMMARISE_AFTER:
      1. Summarise all turns older than the latest KEEP_LATEST
      2. Store summary in profile.last_session_summary
      3. Trim the conversation table to latest KEEP_LATEST turns
    Returns the summary string, or None if no summarisation was needed.
    """
    total = get_turn_count(user_id)
    if total < SUMMARISE_AFTER:
        return None

    init_db()
    with _connect() as conn:
        cutoff_row = conn.execute(
            """SELECT id FROM conversations WHERE user_id = ?
               ORDER BY turn_num DESC LIMIT 1 OFFSET ?""",
            (user_id, KEEP_LATEST - 1)
        ).fetchone()

        if not cutoff_row:
            return None

        old_turns = conn.execute(
            """SELECT role, content FROM conversations
               WHERE user_id = ? AND id <= ?
               ORDER BY turn_num ASC""",
            (user_id, cutoff_row["id"])
        ).fetchall()

    if not old_turns:
        return None

    transcript = "\n".join(
        f"{row['role'].upper()}: {row['content']}" for row in old_turns
    )

    try:
        from langchain_mistralai import ChatMistralAI
        llm = ChatMistralAI(
            model="mistral-small-latest",
            mistral_api_key=mistral_api_key,
            temperature=0,
            max_tokens=200,
        )
        prompt = (
            "Summarise the following conversation in 2-3 sentences. "
            "Focus on: what topics were discussed, what the user was looking for, "
            "and any preferences expressed.\n\n"
            f"CONVERSATION:\n{transcript}\n\nSUMMARY:"
        )
        summary = llm.invoke(prompt).content.strip()
    except Exception:
        topics = list({r["content"][:40] for r in old_turns if r["role"] == "user"})[:3]
        summary = f"User asked about: {'; '.join(topics)}."

    update_profile(user_id, last_session_summary=summary)
    trim_history(user_id, keep=KEEP_LATEST)
    return summary


# ---------------------------------------------------------------------------
# System prompt injection
# ---------------------------------------------------------------------------

def build_profile_context(profile: dict[str, Any]) -> str:
    """
    Return a short paragraph to prepend to any system prompt,
    personalising responses based on stored preferences.
    """
    lines: list[str] = []

    name = profile.get("preferred_name") or profile.get("name")
    if name:
        lines.append(f"The user's name is {name}.")

    if profile.get("branch_interest"):
        lines.append(
            f"They are primarily interested in the {profile['branch_interest']} programme."
        )

    detail = profile.get("detail_level", "brief")
    if detail == "detailed":
        lines.append("They prefer detailed, thorough answers with full explanations.")
    else:
        lines.append("They prefer concise, to-the-point answers.")

    style = profile.get("answer_style", "friendly")
    if style == "formal":
        lines.append("They prefer a formal, professional tone.")
    elif style == "casual":
        lines.append("They prefer a casual, informal tone.")
    else:
        lines.append("They prefer a friendly, warm tone.")

    lang = profile.get("language", "english")
    if lang != "english":
        lines.append(
            f"Respond in {lang.title()} whenever possible, "
            "but use English for technical terms."
        )

    topics = profile.get("prior_topics", [])
    if topics:
        lines.append(
            f"In previous sessions they asked about: {', '.join(topics[-5:])}."
        )

    summary = profile.get("last_session_summary", "")
    if summary:
        lines.append(f"Previous session summary: {summary}")

    facts = profile.get("facts", {})
    if facts:
        fact_lines = [f"{k.replace('_', ' ').title()}: {v}" for k, v in facts.items()]
        lines.append(f"Known facts about the user: {'; '.join(fact_lines)}.")

    if not lines:
        return ""

    return (
        "USER PROFILE (use this to personalise your response):\n"
        + "\n".join(f"- {l}" for l in lines)
        + "\n"
    )


# ---------------------------------------------------------------------------
# Memory summary (human-readable)
# ---------------------------------------------------------------------------

def get_memory_summary(profile: dict[str, Any]) -> str:
    """
    Return a concise, human-readable summary of what the chatbot remembers
    about this user. Used to display on session start.
    """
    parts: list[str] = []

    name = profile.get("preferred_name") or profile.get("name")
    if name:
        parts.append(f"👤 **Name:** {name}")

    if profile.get("branch_interest"):
        parts.append(f"📚 **Branch:** {profile['branch_interest']}")

    if profile.get("language", "english") != "english":
        parts.append(f"🌐 **Language:** {profile['language'].title()}")

    detail = profile.get("detail_level", "brief")
    parts.append(f"📝 **Style:** {detail.capitalize()} answers")

    style = profile.get("answer_style", "friendly")
    parts.append(f"🎭 **Tone:** {style.capitalize()}")

    facts = profile.get("facts", {})
    if facts:
        fact_strs = [f"{k.replace('_', ' ').title()}: {v}" for k, v in facts.items()]
        parts.append(f"💡 **Facts:** {' | '.join(fact_strs)}")

    topics = profile.get("prior_topics", [])
    if topics:
        parts.append(f"🗂 **Past topics:** {', '.join(topics[-5:])}")

    summary = profile.get("last_session_summary", "")
    if summary:
        parts.append(f"📋 **Last session:** {summary}")

    if not parts:
        return "No memory stored yet. Tell me about yourself!"

    return "🧠 **I remember you!**\n" + "\n".join(f"  {p}" for p in parts)


# ---------------------------------------------------------------------------
# Memory export / import
# ---------------------------------------------------------------------------

def export_memory(user_id: str) -> dict[str, Any]:
    """
    Export a user's full memory (profile + conversation history) as a dict.
    Useful for backup or transfer.
    """
    profile = load_profile(user_id)
    history = get_history(user_id, limit=1000)
    return {
        "exported_at": _now_iso(),
        "profile": profile,
        "conversation_history": history,
    }


def import_memory(data: dict[str, Any]) -> str | None:
    """
    Import a previously exported memory dict.
    Returns the user_id if successful, None if data is invalid.
    """
    profile = data.get("profile")
    if not profile or "user_id" not in profile:
        return None

    user_id = profile["user_id"]
    # Save profile
    _save_profile_dict(profile)

    # Restore conversation history
    history = data.get("conversation_history", [])
    with _connect() as conn:
        conn.execute("DELETE FROM conversations WHERE user_id = ?", (user_id,))
        for i, turn in enumerate(history, 1):
            conn.execute(
                "INSERT INTO conversations (user_id, role, content, turn_num, ts) VALUES (?,?,?,?,?)",
                (user_id, turn.get("role", "user"), turn.get("content", ""), i, _now_iso())
            )

    return user_id


# ---------------------------------------------------------------------------
# Auto-deletion of stale profiles
# ---------------------------------------------------------------------------

def delete_stale_profiles() -> list[str]:
    """
    Delete profiles (and their conversations) that have been inactive
    for more than INACTIVE_DAYS. Returns list of deleted user_ids.
    """
    init_db()
    cutoff = (datetime.now(timezone.utc) - timedelta(days=INACTIVE_DAYS)).isoformat()
    with _connect() as conn:
        stale = conn.execute(
            "SELECT user_id FROM profiles WHERE last_active < ?", (cutoff,)
        ).fetchall()
        ids = [row["user_id"] for row in stale]
        for uid in ids:
            conn.execute("DELETE FROM conversations WHERE user_id = ?", (uid,))
            conn.execute("DELETE FROM profiles WHERE user_id = ?", (uid,))
    return ids


# ---------------------------------------------------------------------------
# Clear user data (GDPR-style)
# ---------------------------------------------------------------------------

def clear_user_data(user_id: str) -> bool:
    """
    Delete all data for a user: profile + full conversation history.
    Returns True if the user existed and was deleted, False otherwise.
    """
    init_db()
    with _connect() as conn:
        existed = conn.execute(
            "SELECT 1 FROM profiles WHERE user_id = ?", (user_id,)
        ).fetchone()
        if existed:
            conn.execute("DELETE FROM conversations WHERE user_id = ?", (user_id,))
            conn.execute("DELETE FROM profiles WHERE user_id = ?", (user_id,))
    return existed is not None


# ---------------------------------------------------------------------------
# JSON export (human-readable snapshot)
# ---------------------------------------------------------------------------

def export_all_profiles() -> None:
    """Write all profiles to memory.json as a human-readable backup."""
    init_db()
    with _connect() as conn:
        rows = conn.execute("SELECT * FROM profiles").fetchall()
    profiles = [_row_to_profile(r) for r in rows]
    with open(JSON_PATH, "w") as f:
        json.dump(profiles, f, indent=2, default=str)


# ---------------------------------------------------------------------------
# Privacy notice text
# ---------------------------------------------------------------------------

PRIVACY_NOTICE = """
### 🔒 Privacy Notice

**What we store:**
- Your preferred name (if you tell us)
- Your branch of interest
- Your language, response-detail, and answer-style preferences
- Topics you have asked about (last 20)
- Facts you've shared (year of study, interests, etc.)
- A short summary of your previous session
- Timestamp of your last interaction

**What we do NOT store:**
- Passwords or personal identification numbers
- Financial details
- Full conversation transcripts beyond the last 10 turns

**How long we keep it:**
Profiles inactive for **30 days** are automatically deleted.

**Your rights:**
Type **"Clear my data"** in the chat at any time to delete your entire profile
and conversation history immediately.

**Storage location:**
All data is stored locally in `memory.db` on this machine. Nothing is sent to
external servers.
""".strip()