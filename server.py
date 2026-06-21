from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr
from passlib.context import CryptContext
import sqlite3
import json
from datetime import datetime
from typing import Optional

app = FastAPI(title="SpikeSpeed API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

DB_PATH = "database.db"


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            speed INTEGER NOT NULL,
            reach_height REAL NOT NULL,
            date TEXT NOT NULL,
            hit_coords TEXT NOT NULL,
            land_coords TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)
    conn.commit()
    conn.close()


init_db()


# --- Pydantic Models ---

class RegisterRequest(BaseModel):
    email: str
    password: str


class LoginRequest(BaseModel):
    email: str
    password: str


class HistoryEntry(BaseModel):
    user_id: int
    speed: int
    reach_height: float
    hit_coords: str   # JSON string: {"x": 4.5, "y": 7.0}
    land_coords: str  # JSON string: {"x": 4.5, "y": 13.0}


# --- Endpoints ---

@app.post("/api/register")
def register(data: RegisterRequest):
    if len(data.password) < 6:
        raise HTTPException(status_code=400, detail="Пароль должен быть не менее 6 символов")
    hashed = pwd_context.hash(data.password)
    conn = get_db()
    try:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO users (email, password_hash) VALUES (?, ?)",
            (data.email.lower().strip(), hashed)
        )
        conn.commit()
        user_id = cur.lastrowid
        return {"success": True, "user_id": user_id, "email": data.email.lower().strip()}
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=409, detail="Пользователь с таким email уже существует")
    finally:
        conn.close()


@app.post("/api/login")
def login(data: LoginRequest):
    conn = get_db()
    try:
        cur = conn.cursor()
        cur.execute("SELECT id, email, password_hash FROM users WHERE email = ?", (data.email.lower().strip(),))
        user = cur.fetchone()
        if not user or not pwd_context.verify(data.password, user["password_hash"]):
            raise HTTPException(status_code=401, detail="Неверный email или пароль")
        return {"success": True, "user_id": user["id"], "email": user["email"]}
    finally:
        conn.close()


@app.post("/api/history")
def add_history(entry: HistoryEntry):
    conn = get_db()
    try:
        cur = conn.cursor()
        # Verify user exists
        cur.execute("SELECT id FROM users WHERE id = ?", (entry.user_id,))
        if not cur.fetchone():
            raise HTTPException(status_code=404, detail="Пользователь не найден")
        date_str = datetime.now().strftime("%d.%m.%Y %H:%M")
        cur.execute(
            "INSERT INTO history (user_id, speed, reach_height, date, hit_coords, land_coords) VALUES (?, ?, ?, ?, ?, ?)",
            (entry.user_id, entry.speed, entry.reach_height, date_str, entry.hit_coords, entry.land_coords)
        )
        conn.commit()
        return {"success": True, "id": cur.lastrowid}
    finally:
        conn.close()


@app.get("/api/history/{user_id}")
def get_history(user_id: int):
    conn = get_db()
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT id, speed, reach_height, date, hit_coords, land_coords FROM history WHERE user_id = ? ORDER BY id DESC",
            (user_id,)
        )
        rows = cur.fetchall()
        result = []
        for r in rows:
            result.append({
                "id": r["id"],
                "speed": r["speed"],
                "reach_height": r["reach_height"],
                "date": r["date"],
                "hit_coords": json.loads(r["hit_coords"]),
                "land_coords": json.loads(r["land_coords"]),
            })
        return {"history": result}
    finally:
        conn.close()


@app.get("/api/stats")
def get_stats():
    conn = get_db()
    try:
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) as total, MAX(speed) as max_speed FROM history")
        row = cur.fetchone()
        return {
            "total_spikes": row["total"] or 0,
            "max_speed": row["max_speed"] or 0
        }
    finally:
        conn.close()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
