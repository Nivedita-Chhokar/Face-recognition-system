"""
security_system.py
Main security system: live webcam feed with face recognition.
Displays ENTRY ALLOWED (green) or ENTRY REJECTED (red) on screen.
"""
import cv2
import face_recognition
import numpy as np
import sqlite3
import time
from datetime import datetime
from register_faces import load_all_encodings

DB_PATH = "residents.db"
TOLERANCE = 0.5  # Face matching tolerance (lower = stricter)

def log_access(resident_id, name, status, confidence):
    """Log each access attempt to the database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO access_logs (resident_id, name, status, confidence)
        VALUES (?, ?, ?, ?)
    ''', (resident_id, name, status, confidence))
    conn.commit()
    conn.close()

def get_resident_id(name):
    """Get resident ID from name."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM residents WHERE name = ?", (name,))
    row = cursor.fetchone()
    conn.close()
    return row[0] if row else None

def run_security_system():
    """Main loop: capture webcam, detect faces, compare, display result."""
    print("🔄 Loading registered faces from database...")
    known_names, known_flats, known_encodings = load_all_encodings()

    if len(known_encodings) == 0:
        print("❌ No registered faces found. Please register residents first.")
        return

    print(f"✅ Loaded {len(known_encodings)} face encodings for {len(set(known_names))} residents.")
    print("🎥 Starting security system... Press 'q' to quit.\n")

    cap = cv2.VideoCapture(0)

    # Variables for throttling recognition (for performance)
    process_every_n_frames = 2
    frame_count = 0
    last_recognized_name = None
    last_recognized_time = 0
    cooldown_seconds = 3  # Don't re-log same person within this window

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to read from webcam.")
            break

        frame_count += 1
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Only process every N frames for performance
        if frame_count % process_every_n_frames == 0:
            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

            face_names = []
            face_confidences = []

            for face_encoding in face_encodings:
                # Compare against known encodings
                distances = face_recognition.face_distance(known_encodings, face_encoding)

                if len(distances) > 0:
                    best_match_index = np.argmin(distances)
                    distance = distances[best_match_index]

                    if distance < TOLERANCE:
                        name = known_names[best_match_index]
                        flat = known_flats[best_match_index]
                        confidence = (1 - distance) * 100
                        face_names.append(name)
                        face_confidences.append(confidence)

                        # Log access (with cooldown)
                        current_time = time.time()
                        if (name != last_recognized_name or
                            current_time - last_recognized_time > cooldown_seconds):

                            resident_id = get_resident_id(name)
                            log_access(resident_id, name, "ALLOWED", confidence)
                            print(f"✅ {name} (Flat {flat}) - Entry Allowed "
                                  f"[Confidence: {confidence:.1f}%]")

                            last_recognized_name = name
                            last_recognized_time = current_time
                    else:
                        face_names.append("Unknown")
                        face_confidences.append(0)

                        # Log unknown person
                        current_time = time.time()
                        if (last_recognized_name != "Unknown" or
                            current_time - last_recognized_time > cooldown_seconds):
                            log_access(None, "Unknown", "REJECTED", 0)
                            print(f"❌ Unknown person detected - Entry Rejected")
                            last_recognized_name = "Unknown"
                            last_recognized_time = current_time
                else:
                    face_names.append("Unknown")
                    face_confidences.append(0)

        # Draw results on frame
        if 'face_locations' in locals() and 'face_names' in locals():
            for (top, right, bottom, left), name, conf in zip(
                    face_locations, face_names, face_confidences):

                if name != "Unknown":
                    # GREEN for recognized residents
                    color = (0, 255, 0)
                    label = f"{name} [{conf:.0f}%]"
                    status_text = "ENTRY ALLOWED"
                    status_color = (0, 255, 0)
                else:
                    # RED for unknown
                    color = (0, 0, 255)
                    label = "Unknown"
                    status_text = "ENTRY REJECTED"
                    status_color = (0, 0, 255)

                # Draw face box
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                # Draw label below face
                cv2.rectangle(frame, (left, bottom - 25), (right, bottom), color, cv2.FILLED)
                cv2.putText(frame, label, (left + 6, bottom - 6),
                            cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)

                # Draw large status text at top
                cv2.rectangle(frame, (0, 0), (frame.shape[1], 60), status_color, cv2.FILLED)
                cv2.putText(frame, status_text, (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

        # Timestamp overlay
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, timestamp, (10, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow("Smart Society Security System", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("\n🛑 Security system stopped.")

if __name__ == "__main__":
    print("=" * 55)
    print("  🏘️  Smart Society Security System - Face Recognition")
    print("=" * 55)
    run_security_system()