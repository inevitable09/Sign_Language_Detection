import os
import cv2
import numpy as np

# Create the dataset directory if it doesn't exist
DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

number_of_classes = 28  # Number of different sign language classes
dataset_size = 200     # Number of images per class

# OpenCV video capture (ensure you're using the correct camera index)
cap = cv2.VideoCapture(0)  # Change the index if needed (0 is usually the default webcam)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Create class subdirectories
for j in range(number_of_classes):
    class_dir = os.path.join(DATA_DIR, str(j))
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    print(f'Collecting data for class {j}')

    # Wait for user to press 'Q' to start collecting images
    done = False
    while not done:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame.")
            break
        
        cv2.putText(frame, 'Ready? Press "Q" to start collecting images', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
        cv2.imshow('frame', frame)
        
        # Break if 'Q' is pressed to start image collection
        if cv2.waitKey(25) == ord('q'):
            done = True

    # Start collecting images for the current class
    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame.")
            break
        
        # Save image to the corresponding class folder
        image_path = os.path.join(DATA_DIR, str(j), f'{counter}.jpg')
        cv2.imwrite(image_path, frame)
        cv2.imshow('frame', frame)

        # Show progress
        print(f'Collecting image {counter + 1}/{dataset_size} for class {j}')
        
        # Wait for a short period before capturing the next frame
        cv2.waitKey(25)
        counter += 1

# Release video capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()

print("Data collection complete.")
