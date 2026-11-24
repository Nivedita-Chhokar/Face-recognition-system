"""
register_faces.py
Captures face images via webcam, encodes them, and stores in the database.
"""
import cv2
import face_recognition
import sqlite3
import numpy as np
import os
from datetime import datetime

DB_PATH = "residents.db"
IMAGE_DIR = "registered_faces"

def get_db_connection():
    return sqlite3.connect(DB_PATH)

def register_resident():
    """Interactive flow to register a new resident with face encoding."""
    os.makedirs(IMAGE_DIR, exist_ok=True)

    name = input("Enter resident's full name: ").strip()
    flat_number = input("Enter flat number (e.g., A-101): ").strip()
    phone = input("Enter phone number: ").strip()

    conn = get_db_connection()
    cursor = conn.cursor()

    # Insert resident details
    cursor.execute('''
        INSERT INTO residents (name, flat_number, phone)
        VALUES (?, ?, ?)
    ''', (name, flat_number, phone))
    resident_id = cursor.lastrowid

    print(f"\n📸 Capturing face for {name}. Look at the camera.")
    print("   Press 'c' to capture, 'q' to quit.\n")

    cap = cv2.VideoCapture(0)
    captured_count = 0
    encodings = []

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to access webcam.")
            break

        # Convert BGR to RGB for face_recognition
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect faces
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        # Draw rectangles around detected faces
        for (top, right, bottom, left) in face_locations:
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        # Show status
        status_text = f"Captured: {captured_count}/5"
        cv2.putText(frame, status_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, "Press 'c' to capture, 'q' to quit", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        cv2.imshow("Register Face", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        if key == ord('c') and len(face_encodings) > 0:
            captured_count += 1
            # Save the image
            img_path = os.path.join(IMAGE_DIR, f"{name.replace(' ', '_')}_{captured_count}.jpg")
            cv2.imwrite(img_path, frame)

            # Save encoding to DB
            encoding_str = ','.join(map(str, face_encodings[0]))
            cursor.execute('''
                INSERT INTO face_encodings (resident_id, encoding, image_path)
                VALUES (?, ?, ?)
            ''', (resident_id, encoding_str, img_path))
            encodings.append(face_encodings[0])

            print(f"   ✅ Captured image {captured_count}/5")

            if captured_count >= 5:
                print("   ✔️ 5 images captured. Registration complete!")
                break

    cap.release()
    cv2.destroyAllWindows()
    conn.commit()
    conn.close()

    if captured_count > 0:
        print(f"\n🎉 Resident '{name}' registered successfully with {captured_count} face samples.")
    else:
        print("\n⚠️  No faces captured. Registration cancelled.")

def load_all_encodings():
    """Helper: load all stored encodings from DB (used by security_system.py)."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        SELECT r.name, r.flat_number, f.encoding
        FROM face_encodings f
        JOIN residents r ON f.resident_id = r.id
    ''')
    rows = cursor.fetchall()
    conn.close()

    known_names = []
    known_flats = []
    known_encodings = []

    for name, flat, enc_str in rows:
        encoding = np.array([float(x) for x in enc_str.split(',')])
        known_names.append(name)
        known_flats.append(flat)
        known_encodings.append(encoding)

    return known_names, known_flats, known_encodings

if __name__ == "__main__":
    print("=" * 50)
    print("  Smart Society - Face Registration System")
    print("=" * 50)
    register_resident()