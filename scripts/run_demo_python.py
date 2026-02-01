#!/usr/bin/env python3
"""Demo script that creates a buggy Python/FastAPI repo and runs the daydream reviewer on it.

Usage:
    python scripts/run_demo_python.py [DIRECTORY] [--cleanup]

Arguments:
    DIRECTORY    Where to create the test repo (default: ../test_buggy_demo)

Options:
    --cleanup    Remove the test repo after running
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

# Default test repo location (sibling to daydream)
DEFAULT_REPO_PATH = Path(__file__).parent.parent.parent / "test_buggy_demo"

# Sample buggy Python files with intentional issues for the reviewer to find

MAIN_PY = '''"""FastAPI application with intentional bugs for testing the reviewer."""
from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
import json  # unused import
import os  # unused import

from database import get_db, init_db, User, Note, search_notes_unsafe
from models import UserCreate, UserResponse, NoteCreate, NoteResponse, NoteUpdate
from utils import process_items, log_action

app = FastAPI()

init_db()


@app.post("/users/", response_model=UserResponse)
def create_user(user: UserCreate, db: Session = Depends(get_db)):
    # BUG: storing plaintext password
    db_user = User(username=user.username, email=user.email, password=user.password)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    print(f"Created user: {user.username}")  # BUG: print instead of logging
    return db_user


@app.get("/users/{user_id}", response_model=UserResponse)
def get_user(user_id: int, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.id == user_id).first()
    return user  # BUG: no 404 handling if user not found


@app.post("/notes/", response_model=NoteResponse)
def create_note(note: NoteCreate, user_id: int, db: Session = Depends(get_db)):
    # BUG: no validation that user_id exists
    db_note = Note(
        title=note.title,
        content=note.content,
        priority=note.priority,
        user_id=user_id
    )
    db.add(db_note)
    db.commit()
    log_action("create_note", user_id)
    return db_note


@app.get("/notes/", response_model=List[NoteResponse])
def get_all_notes(db: Session = Depends(get_db)):
    notes = db.query(Note).all()
    # BUG: N+1 query pattern
    for note in notes:
        _ = note.owner.username  # triggers lazy load for each note
    return notes


@app.get("/notes/search/")
def search_notes(query: str, db: Session = Depends(get_db)):
    # BUG: uses unsafe SQL query function
    try:
        results = search_notes_unsafe(db, query)
        return {"results": results}
    except Exception as e:  # BUG: broad exception, unused variable
        return {"error": "Search failed"}


@app.put("/notes/{note_id}")
def update_note(note_id, update: NoteUpdate, db: Session = Depends(get_db)):
    # BUG: note_id has no type hint
    note = db.query(Note).filter(Note.id == note_id).first()
    if update.title:
        note.title = update.title  # BUG: no null check on note
    if update.content:
        note.content = update.content
    if update.priority:
        note.priority = update.priority
    db.commit()
    return {"status": "updated"}


@app.delete("/notes/{note_id}")
async def delete_note(note_id: int, db: Session = Depends(get_db)):
    note = db.query(Note).filter(Note.id == note_id).first()
    db.delete(note)  # BUG: no check if note exists
    db.commit()
    return {"deleted": note_id}
'''

DATABASE_PY = '''"""Database models and connection handling."""
from sqlalchemy import create_engine, Column, Integer, String, ForeignKey, Text, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship

DATABASE_URL = "sqlite:///./notes.db"
SECRET_KEY = "hardcoded-secret-key-12345"  # BUG: hardcoded secret

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True)
    email = Column(String(100))
    password = Column(String(100))  # BUG: plaintext password storage
    notes = relationship("Note", back_populates="owner")


class Note(Base):
    __tablename__ = "notes"
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(100))
    content = Column(Text)
    priority = Column(Integer, default=1)
    user_id = Column(Integer, ForeignKey("users.id"))
    owner = relationship("User", back_populates="notes")


def init_db():
    Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def search_notes_unsafe(db, search_term):
    # BUG: SQL injection vulnerability
    query = f"SELECT * FROM notes WHERE title LIKE '%{search_term}%'"
    result = db.execute(text(query))
    return result.fetchall()
'''

MODELS_PY = '''"""Pydantic models for request/response validation."""
from pydantic import BaseModel
from typing import Optional
import re  # unused import


class UserCreate(BaseModel):
    username: str
    email: str
    password: str  # BUG: no password strength validation


class UserResponse(BaseModel):
    id: int
    username: str
    email: str

    class Config:
        from_attributes = True


class NoteCreate(BaseModel):
    title: str
    content: str
    priority: int  # BUG: no validation on range


class NoteResponse(BaseModel):
    id: int
    title: str
    content: str
    priority: int
    user_id: int

    class Config:
        from_attributes = True


class NoteUpdate(BaseModel):
    title: Optional[str] = None
    content: Optional[str] = None
    priority: Optional[int] = None


unused_constant = "this is never used"  # BUG: dead code


def validate_email(email):  # BUG: no type hints
    if "@" in email:
        return True
    return False
'''

UTILS_PY = '''"""Utility functions."""
import datetime
import random
import string
import time  # unused import


def generate_token(length=32):
    # BUG: using random instead of secrets for security token
    chars = string.ascii_letters + string.digits
    return ''.join(random.choice(chars) for _ in range(length))


def process_items(items=[]):  # BUG: mutable default argument
    items.append("processed")
    return items


def calculate_priority_score(priority, age_days):
    # BUG: magic numbers without constants
    if priority > 5:
        return priority * 1.5 + age_days * 0.1
    elif priority > 3:
        return priority * 1.2 + age_days * 0.05
    else:
        return priority * 1.0 + age_days * 0.02


def format_note_summary(note):  # BUG: no type hints
    try:
        return f"{note.title[:20]}... - Priority: {note.priority}"
    except Exception:  # BUG: broad exception catch
        return "Error formatting note"


def log_action(action, user_id, details=[]):  # BUG: mutable default argument
    print(f"[{datetime.datetime.now()}] User {user_id}: {action}")  # BUG: print instead of logging
    details.append(action)
    return details


async def fetch_external_data(url):
    # BUG: doesn't actually await anything
    import aiohttp
    session = aiohttp.ClientSession()
    response = session.get(url)  # BUG: missing await
    return response


def sanitize_input(text):
    # BUG: incomplete sanitization (XSS vulnerability)
    return text.replace("<", "").replace(">", "")
'''


def create_test_repo(repo_path: Path) -> Path | None:
    """Create the test repo with buggy files.

    Returns None if user declines to overwrite existing directory.
    """
    print(f"Creating test repo at: {repo_path}")

    # Ask for confirmation if exists
    if repo_path.exists():
        response = input(f"\nDirectory {repo_path} already exists. Overwrite? [y/N] ")
        if response.lower() != "y":
            print("Aborted.")
            return None
        shutil.rmtree(repo_path)

    repo_path.mkdir(parents=True)

    # Write the buggy files
    files = {
        "main.py": MAIN_PY,
        "database.py": DATABASE_PY,
        "models.py": MODELS_PY,
        "utils.py": UTILS_PY,
    }

    for filename, content in files.items():
        filepath = repo_path / filename
        filepath.write_text(content)
        print(f"  Created: {filename}")

    # Initialize git repo
    subprocess.run(["git", "init"], cwd=repo_path, capture_output=True)
    subprocess.run(["git", "add", "."], cwd=repo_path, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "Initial commit with buggy code"],
        cwd=repo_path,
        capture_output=True,
    )
    print("  Initialized git repo")

    return repo_path


def run_daydream(target: Path) -> int:
    """Run daydream on the target repo."""
    print(f"\nRunning daydream on: {target}")
    print("-" * 60)

    result = subprocess.run(
        ["python", "-m", "daydream", str(target), "--python", "--debug", "--no-cleanup"],
        cwd=Path(__file__).parent.parent,
    )

    return result.returncode


def cleanup_test_repo(repo_path: Path):
    """Remove the test repo."""
    if repo_path.exists():
        print(f"\nCleaning up: {repo_path}")
        shutil.rmtree(repo_path)


def main():
    parser = argparse.ArgumentParser(description="Run daydream demo on a buggy test repo")
    parser.add_argument(
        "directory",
        nargs="?",
        type=Path,
        default=DEFAULT_REPO_PATH,
        help=f"Where to create the test repo (default: {DEFAULT_REPO_PATH})",
    )
    parser.add_argument("--cleanup", action="store_true", help="Remove test repo after running")
    args = parser.parse_args()

    repo_path = args.directory.resolve()

    try:
        # Create test repo
        target = create_test_repo(repo_path)
        if target is None:
            return 1

        # Run daydream
        exit_code = run_daydream(target)

        # Cleanup if requested
        if args.cleanup:
            cleanup_test_repo(repo_path)
        else:
            print(f"\nTest repo preserved at: {repo_path}")

        return exit_code

    except KeyboardInterrupt:
        print("\nInterrupted by user")
        return 130


if __name__ == "__main__":
    sys.exit(main())
