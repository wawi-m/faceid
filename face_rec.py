import cv2
import os
import numpy as np

faces = []  # List to hold faces detected in images
ids = []    # List to hold labels (IDs corresponding to the faces)

# Define image directory
img_dir = "path/to/images"  # Replace with the correct path to your image folder

# Initialize face recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Iterate through directories in the img_dir
for face_id in os.listdir(img_dir):
    face_path = os.path.join(img_dir, face_id)
    
    if os.path.isdir(face_path):  # Check if it's a directory (person's folder)
        for img_label in os.listdir(face_path):
            img_path = os.path.join(face_path, img_label)
            
            img = cv2.imread(img_path)
            
            # Convert to grayscale for face detection
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Detect faces in the image
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces_detected = face_cascade.detectMultiScale(gray_img, 1.3, 5)
            
            # Process each face detected
            for (x, y, w, h) in faces_detected:
                face = gray_img[y:y+h, x:x+w]  # Extract face region
                faces.append(face)  # Add the face to the list
                ids.append(int(face_id))  # Add the corresponding label (ID)
                
# Train the recognizer with the collected faces and labels
recognizer.train(faces, np.array(ids))

# Save the trained model
recognizer.save('trained_model.yml')
