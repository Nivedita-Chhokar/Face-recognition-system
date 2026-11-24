"""
evaluate_system.py
Tests the face recognition system and generates an accuracy report.
"""
import cv2
import face_recognition
import numpy as np
import sqlite3
import os
from register_faces import load_all_encodings

DB_PATH = "residents.db"
TEST_DIR = "test_images"
TOLERANCE = 0.5

def evaluate_system():
    """Run test images through the system and compute accuracy metrics."""
    if not os.path.exists(TEST_DIR):
        print(f"❌ Create a folder '{TEST_DIR}' with subfolders for each person.")
        print("   Structure: test_images/<person_name>/image1.jpg, image2.jpg...")
        return

    known_names, known_flats, known_encodings = load_all_encodings()

    if len(known_encodings) == 0:
        print("❌ No registered faces in database.")
        return

    # Metrics
    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0
    total_tests = 0

    results = []

    for person_folder in os.listdir(TEST_DIR):
        person_path = os.path.join(TEST_DIR, person_folder)
        if not os.path.isdir(person_path):
            continue

        for img_file in os.listdir(person_path):
            img_path = os.path.join(person_path, img_file)
            if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue

            total_tests += 1
            image = face_recognition.load_image_file(img_path)
            face_encodings = face_recognition.face_encodings(image)

            if len(face_encodings) == 0:
                print(f"  ⚠️  No face detected in {img_file}")
                continue

            # Check if this is a known person (folder name matches a resident)
            is_known = person_folder.lower() in [n.lower() for n in known_names]
            expected = "KNOWN" if is_known else "UNKNOWN"

            # Try to match
            distances = face_recognition.face_distance(known_encodings, face_encodings[0])
            best_match_index = np.argmin(distances)
            best_distance = distances[best_match_index]

            if best_distance < TOLERANCE:
                predicted = "KNOWN"
                predicted_name = known_names[best_match_index]
                confidence = (1 - best_distance) * 100
            else:
                predicted = "UNKNOWN"
                predicted_name = "Unknown"
                confidence = 0

            # Determine TP, FP, TN, FN
            if expected == "KNOWN" and predicted == "KNOWN":
                true_positives += 1
                result = "TP ✅"
            elif expected == "KNOWN" and predicted == "UNKNOWN":
                false_negatives += 1
                result = "FN ❌ (Missed known person)"
            elif expected == "UNKNOWN" and predicted == "KNOWN":
                false_positives += 1
                result = "FP ❌ (False match)"
            else:  # UNKNOWN & UNKNOWN
                true_negatives += 1
                result = "TN ✅"

            results.append({
                'file': img_file,
                'expected': expected,
                'predicted': predicted,
                'name': predicted_name,
                'confidence': confidence,
                'result': result
            })

            print(f"  {img_file}: Expected={expected}, Predicted={predicted_name} "
                  f"({confidence:.1f}%) -> {result}")

    # Calculate metrics
    print("\n" + "=" * 60)
    print("  📊 ACCURACY REPORT")
    print("=" * 60)
    print(f"  Total test images:        {total_tests}")
    print(f"  True Positives (TP):      {true_positives}")
    print(f"  True Negatives (TN):      {true_negatives}")
    print(f"  False Positives (FP):     {false_positives}")
    print(f"  False Negatives (FN):     {false_negatives}")
    print("-" * 60)

    if (true_positives + false_positives) > 0:
        precision = true_positives / (true_positives + false_positives) * 100
        print(f"  Precision:                {precision:.2f}%")
    else:
        precision = 0
        print(f"  Precision:                N/A")

    if (true_positives + false_negatives) > 0:
        recall = true_positives / (true_positives + false_negatives) * 100
        print(f"  Recall:                   {recall:.2f}%")
    else:
        recall = 0
        print(f"  Recall:                   N/A")

    if (precision + recall) > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
        print(f"  F1 Score:                 {f1:.2f}%")
    else:
        f1 = 0
        print(f"  F1 Score:                 N/A")

    accuracy = (true_positives + true_negatives) / max(total_tests, 1) * 100
    print(f"  Overall Accuracy:         {accuracy:.2f}%")
    print("=" * 60)

    # Save report to file
    with open("accuracy_report.txt", "w") as f:
        f.write("SMART SOCIETY SECURITY SYSTEM - ACCURACY REPORT\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Total test images:    {total_tests}\n")
        f.write(f"True Positives (TP):  {true_positives}\n")
        f.write(f"True Negatives (TN):  {true_negatives}\n")
        f.write(f"False Positives (FP): {false_positives}\n")
        f.write(f"False Negatives (FN): {false_negatives}\n\n")
        f.write(f"Precision:            {precision:.2f}%\n")
        f.write(f"Recall:               {recall:.2f}%\n")
        f.write(f"F1 Score:             {f1:.2f}%\n")
        f.write(f"Overall Accuracy:     {accuracy:.2f}%\n\n")
        f.write("-" * 60 + "\n")
        f.write("DETAILED RESULTS:\n")
        f.write("-" * 60 + "\n")
        for r in results:
            f.write(f"{r['file']:30s} | Expected: {r['expected']:7s} | "
                    f"Predicted: {r['name']:15s} ({r['confidence']:5.1f}%) | {r['result']}\n")

    print("\n📄 Report saved to 'accuracy_report.txt'")

if __name__ == "__main__":
    evaluate_system()