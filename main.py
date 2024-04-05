import cv2
import numpy as np
import face_recognition
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
import joblib

def load_images(path):
    images = []
    labels = []
    mylist = os.listdir(path)
    for img in mylist:
        curImg = cv2.imread(f'{path}/{img}')
        images.append(curImg)
        labels.append(os.path.splitext(img)[0])
    return images, labels

dataset_path = 'student_image'
images, labels = load_images(dataset_path)

def findEncodings(images, labels):
    encodeList = []
    new_labels = []
    
    for img, label in zip(images, labels):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Detect faces in the image
        face_locations = face_recognition.face_locations(img)
        
        # If no face is detected, skip this image
        if not face_locations:
            print(f"No face found in image, skipping...{label}")
            continue
        
        # Get the face encoding
        encoded_face = face_recognition.face_encodings(img, face_locations)[0]
        encodeList.append(encoded_face)
        new_labels.append(label)
    
    return encodeList, new_labels

encoded_face_train, new_labels_train = findEncodings(images, labels)

# Convert labels to numeric representation
le = LabelEncoder()
encoded_labels = le.fit_transform(new_labels_train)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(encoded_face_train, encoded_labels, test_size=0.2, random_state=42)

# Define the TensorFlow model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(128,)),  # Assuming face encodings are of shape (128,)
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(len(le.classes_), activation='softmax')  # Output layer with softmax activation for classification
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(np.array(encoded_face_train), np.array(encoded_labels), epochs=1000, verbose=0, validation_data=(np.array(X_test), np.array(y_test)))

# Evaluate the model
test_loss, test_accuracy = model.evaluate(np.array(X_test), np.array(y_test))
print("Test Accuracy:", test_accuracy)

# Real-time face recognition using the trained model
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    
    face_locations = face_recognition.face_locations(frame)
    if len(face_locations) > 0:
        face_encodings = face_recognition.face_encodings(frame, face_locations)
        
        # Predict labels using the trained classifier
        labels = model.predict(np.array(face_encodings))
        
        # Convert labels to 1-dimensional array
        labels = labels.flatten()
        # Inverse transform the labels
        names = le.inverse_transform(labels)
        
        
        for (top, right, bottom, left), name in zip(face_locations, names):
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

    cv2.imshow('Face Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
