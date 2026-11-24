"""
database_setup.py
Creates the SQLite database with tables for residents and their face encodings.
"""
import sqlite3
import os

DB_PATH = "residents.db"

def create_database():
    """Creates the residents database with required tables."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Table for resident details
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS residents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            flat_number TEXT NOT NULL,
            phone TEXT,
            registration_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # Table for face encodings (128-dim vectors stored as text)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS face_encodings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            resident_id INTEGER NOT NULL,
            encoding TEXT NOT NULL,
            image_path TEXT,
            FOREIGN KEY (resident_id) REFERENCES residents(id) ON DELETE CASCADE
        )
    ''')

    # Table for access logs
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS access_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            resident_id INTEGER,
            name TEXT,
            status TEXT NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            confidence REAL
        )
    ''')

    conn.commit()
    conn.close()
    print(f"✅ Database '{DB_PATH}' created successfully.")

if __name__ == "__main__":
    create_database()