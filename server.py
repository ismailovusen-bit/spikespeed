from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr
from passlib.context import CryptContext
import sqlite3
import json
import os
import httpx
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

# Groq API key is read from an environment variable -- NEVER hardcode it here,
# since this file lives in a public GitHub repo. Set GROQ_API_KEY on Railway instead.
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "openai/gpt-oss-120b"


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


class AnalyzeRequest(BaseModel):
    speed: int
    reach_height: float
    net_height: float
    flight_time: float
    horizontal_distance: float
    hit_x: float
    hit_y: float
    land_x: float
    land_y: float


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


@app.post("/api/analyze")
def analyze_shot(data: AnalyzeRequest):
    print(f"[Groq] key loaded: {'yes, starts with ' + GROQ_API_KEY[:7] if GROQ_API_KEY else 'NO KEY FOUND'}")
    if not GROQ_API_KEY:
        raise HTTPException(status_code=503, detail="ИИ-анализ временно недоступен (ключ не настроен на сервере)")

    # Where the ball landed relative to the court zones, in plain Russian, for the AI to reason about
    land_zone = "в аут (за пределами площадки)" if not (0 <= data.land_x <= 9 and 9 <= data.land_y <= 18) else "в площадку соперника"

    prompt = (
        f"Ты — опытный тренер по волейболу. Проанализируй один удар (нападающий удар/спайк) игрока "
        f"по следующим измеренным данным и дай короткий практический совет по технике на русском языке.\n\n"
        f"Данные удара:\n"
        f"- Скорость мяча: {data.speed} км/ч\n"
        f"- Высота точки удара (съём): {data.reach_height:.2f} м\n"
        f"- Высота сетки: {data.net_height:.2f} м\n"
        f"- Время полёта мяча: {data.flight_time:.3f} с\n"
        f"- Горизонтальная дистанция полёта: {data.horizontal_distance:.2f} м\n"
        f"- Точка удара на площадке: X={data.hit_x:.1f} Y={data.hit_y:.1f}\n"
        f"- Точка приземления: X={data.land_x:.1f} Y={data.land_y:.1f} ({land_zone})\n\n"
        f"Дай ответ в 3-4 коротких предложениях: оцени силу удара относительно общего уровня игроков, "
        f"прокомментируй траекторию и точку приземления, и дай один конкретный совет по технике для следующего удара. "
        f"Пиши тепло и подбадривающе, но честно. Не используй markdown-разметку, только обычный текст."
    )

    payload = {
        "model": GROQ_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "max_tokens": 300
    }

    try:
        resp = httpx.post(
            GROQ_API_URL,
            json=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {GROQ_API_KEY}"
            },
            timeout=25.0
        )
        print(f"[Groq] status={resp.status_code} body={resp.text[:500]}")

        if resp.status_code != 200:
            raise HTTPException(
                status_code=502,
                detail=f"Groq вернул ошибку {resp.status_code}: {resp.text[:200]}"
            )

        result = resp.json()
        advice = result["choices"][0]["message"]["content"].strip()
        return {"success": True, "advice": advice}

    except httpx.RequestError as e:
        print(f"[Groq] connection error: {repr(e)}")
        raise HTTPException(status_code=502, detail=f"Не удалось связаться с Groq: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
