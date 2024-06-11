import cv2
import sqlite3
from keras.models import load_model
import numpy as np
from threading import Thread

# Load pre-trained face recognition model
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read("models/trained_lbph_face_recognizer_model.yml")

# Load Haarcascade for face detection
face_cascade = cv2.CascadeClassifier("models/haarcascade_frontalface_default.xml")

# Font settings for display
font_face = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.6
font_color = (255, 255, 255)
font_thickness = 2
font_bottom_margin = 30

# Nametag settings
nametag_color = (100, 180, 0)
nametag_height = 50

# Face rectangle settings
face_rectangle_color = nametag_color
face_rectangle_thickness = 2

# Load gesture recognition model
gesture_model = load_model("./models/model.h5", compile=False)
gesture_labels = ["ok", "none"]

# Open a connection to the first webcam
camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

def handle_purchase(frame, customer_name,):
    conn = sqlite3.connect("customer_data.db")
    cursor = conn.cursor()
    resized_image = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
    normalized_image = (resized_image.astype(np.float32).reshape(1, 224, 224, 3) / 127.5) - 1
    prediction = gesture_model.predict(normalized_image)
    predicted_index = np.argmax(prediction)
    label = gesture_labels[predicted_index]
    confidence_score = prediction[0][predicted_index]

    cursor.execute('''CREATE TABLE IF NOT EXISTS purchases (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        customer_name TEXT,
                        bought_item TEXT
                    )''')
    if label == "ok" and confidence_score > 0.75:
        cursor.execute("INSERT INTO purchases (customer_name, bought_item) VALUES (?, ?)", (customer_name, "OK"))
        conn.commit()
        print(f"{customer_name} bought an item")
    else:
        conn.execute('''SELECT * FROM purchases''')
        rows = cursor.fetchmany()
        print(f"{customer_name} did not buy an item. Total purchases: {len(rows)}")

def main():
    try:
        conn = sqlite3.connect("customer_data.db")
        cursor = conn.cursor()
        print("Successfully connected to the database")
    except sqlite3.Error as e:
        print("SQLite error:", e)
        return

    while True:
        ret, frame = camera.read()
        if not ret:
            break

        # Convert frame to grayscale for face detection
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        detected_faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in detected_faces:
            customer_uid, confidence = face_recognizer.predict(gray_frame[y:y+h, x:x+w])
            cursor.execute("SELECT customer_name FROM customers WHERE customer_uid = ?", (customer_uid,))
            row = cursor.fetchone()
            customer_name = row[0].split(" ")[0] if row else "Unknown"

            if 45 < confidence < 100:
                purchase_thread = Thread(target=handle_purchase, args=(frame, customer_name))
                purchase_thread.start()

                # Draw rectangle around the face
                cv2.rectangle(frame, (x-20, y-20), (x+w+20, y+h+20), face_rectangle_color, face_rectangle_thickness)

                # Display name tag
                cv2.rectangle(frame, (x-22, y-nametag_height), (x+w+22, y-22), nametag_color, -1)
                cv2.putText(frame, f"{customer_name}: {round(confidence, 2)}%", (x, y-font_bottom_margin), font_face, font_scale, font_color, font_thickness)

        # Display the resulting frame
        cv2.imshow("Detecting Faces...", frame)

        # Exit loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Release the camera and close all OpenCV windows
    camera.release()
    cv2.destroyAllWindows()

    # Close the database connection
    conn.close()

if __name__ == "__main__":
    main()
