import sqlite3
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

DB = "submissions.db"

def get_db():
    con = sqlite3.connect(DB)
    con.row_factory = sqlite3.Row
    return con

@asynccontextmanager
async def lifespan(app):
    con = get_db()
    con.execute(
        """CREATE TABLE IF NOT EXISTS submissions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            name TEXT,
            email TEXT,
            phone TEXT,
            address TEXT,
            message TEXT
        )"""
    )
    con.commit()
    con.close()
    yield

app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class SubmissionIn(BaseModel):
    username: str
    name: str
    email: str
    phone: str
    address: str
    message: str

@app.post("/api/submissions")
def add_submission(s: SubmissionIn):
    con = get_db()
    con.execute(
        "INSERT INTO submissions (username,name,email,phone,address,message) VALUES (?,?,?,?,?,?)",
        (s.username, s.name, s.email, s.phone, s.address, s.message),
    )
    con.commit()
    con.close()
    return {"ok": True}

@app.get("/api/submissions")
def get_submissions(username: str):
    con = get_db()
    rows = con.execute(
        "SELECT name,email,phone,address,message FROM submissions WHERE username=?",
        (username,),
    ).fetchall()
    con.close()
    return [dict(r) for r in rows]

if Path("dist").is_dir():
    app.mount("/", StaticFiles(directory="dist", html=True), name="static")
