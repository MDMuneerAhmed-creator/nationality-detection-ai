import sqlite3
import os
from typing import Dict, List, Any, Optional
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "database.db")

def get_connection():
    os.makedirs(os.path.dirname(os.path.abspath(DB_PATH)), exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_connection()
    cursor = conn.cursor()
    
    # Table for prediction logs
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        filename TEXT NOT NULL,
        image_path TEXT,
        nationality TEXT NOT NULL,
        nationality_confidence REAL NOT NULL,
        emotion TEXT NOT NULL,
        emotion_confidence REAL NOT NULL,
        age TEXT,
        age_confidence REAL,
        dress_color TEXT,
        dress_color_confidence REAL,
        conditional_rule TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)
    
    # Table for model evaluation metrics
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS model_metrics (
        model_name TEXT PRIMARY KEY,
        accuracy REAL NOT NULL,
        precision REAL NOT NULL,
        recall REAL NOT NULL,
        f1_score REAL NOT NULL,
        loss REAL NOT NULL,
        epochs_trained INTEGER NOT NULL,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    # Table for training history per epoch
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS training_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        model_name TEXT NOT NULL,
        epoch INTEGER NOT NULL,
        train_acc REAL NOT NULL,
        val_acc REAL NOT NULL,
        train_loss REAL NOT NULL,
        val_loss REAL NOT NULL,
        lr REAL NOT NULL,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)
    
    conn.commit()
    conn.close()

def save_prediction(
    filename: str,
    image_path: str,
    nationality: str,
    nationality_confidence: float,
    emotion: str,
    emotion_confidence: float,
    age: Optional[str] = None,
    age_confidence: Optional[float] = None,
    dress_color: Optional[str] = None,
    dress_color_confidence: Optional[float] = None,
    conditional_rule: str = ""
) -> int:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
    INSERT INTO predictions (
        filename, image_path, nationality, nationality_confidence,
        emotion, emotion_confidence, age, age_confidence,
        dress_color, dress_color_confidence, conditional_rule
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        filename, image_path, nationality, nationality_confidence,
        emotion, emotion_confidence, age, age_confidence,
        dress_color, dress_color_confidence, conditional_rule
    ))
    prediction_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return prediction_id

def get_recent_predictions(limit: int = 20) -> List[Dict[str, Any]]:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
    SELECT * FROM predictions ORDER BY id DESC LIMIT ?
    """, (limit,))
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]

def save_model_metrics(
    model_name: str,
    accuracy: float,
    precision: float,
    recall: float,
    f1_score: float,
    loss: float,
    epochs_trained: int
):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
    INSERT INTO model_metrics (model_name, accuracy, precision, recall, f1_score, loss, epochs_trained, updated_at)
    VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
    ON CONFLICT(model_name) DO UPDATE SET
        accuracy=excluded.accuracy,
        precision=excluded.precision,
        recall=excluded.recall,
        f1_score=excluded.f1_score,
        loss=excluded.loss,
        epochs_trained=excluded.epochs_trained,
        updated_at=CURRENT_TIMESTAMP
    """, (model_name, accuracy, precision, recall, f1_score, loss, epochs_trained))
    conn.commit()
    conn.close()

def save_epoch_history(
    model_name: str,
    epoch: int,
    train_acc: float,
    val_acc: float,
    train_loss: float,
    val_loss: float,
    lr: float
):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
    INSERT INTO training_history (model_name, epoch, train_acc, val_acc, train_loss, val_loss, lr)
    VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (model_name, epoch, train_acc, val_acc, train_loss, val_loss, lr))
    conn.commit()
    conn.close()

def get_all_metrics() -> List[Dict[str, Any]]:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM model_metrics")
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]

def get_model_history(model_name: str) -> List[Dict[str, Any]]:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM training_history WHERE model_name=? ORDER BY epoch ASC", (model_name,))
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]

if __name__ == "__main__":
    init_db()
    print("Database initialized successfully.")
