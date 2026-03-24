import pickle
import cv2
import mediapipe as mp
import numpy as np

# Load the trained model (with exception handling)
try:
    model_dict = pickle.load(open('./model3_fixed.pickle', 'rb'))
    model = model_dict['model']
except FileNotFoundError:
    print("Error: Model file not found!")
    exit()

cap = cv2.VideoCapture(0)  # Default webcam
if not cap.isOpened():
    print("Error: Couldn't access the webcam.")
    exit()

# Mediapipe hands module setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.3, min_tracking_confidence=0.3)

# Labels dictionary (for A-Z, Del, Space) → 0-27
labels_dict = {i: chr(65 + i) for i in range(26)}  # 0 -> 'A', ..., 25 -> 'Z'
labels_dict[26] = 'Space'
labels_dict[27] = 'Del'

while True:
    data_aux = []  # This will hold the features for prediction
    x_ = []  # Temporary list to store x coordinates
    y_ = []  # Temporary list to store y coordinates

    ret, frame = cap.read()
    if not ret:
        print("Error: Couldn't read frame.")
        break

    H, W, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Drawing landmarks and connections on the frame
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

            # Extract x and y coordinates of landmarks
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                x_.append(x)
                y_.append(y)

            # Normalize and append to data_aux (feature vector)
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))  # Normalize x coordinates
                data_aux.append(y - min(y_))  # Normalize y coordinates

        # Make prediction based on features and get confidence
        try:
            prediction_probabilities = model.predict_proba([np.asarray(data_aux)])  # Get probability distribution
            print(2)
            predicted_class_index = np.argmax(prediction_probabilities)  # Get the index of the highest probability
            print(3)
            confidence = prediction_probabilities[0][predicted_class_index] * 100  # Get confidence level
            print(4)
            predicted_character = labels_dict[int(predicted_class_index)]
            print(5)
        except Exception as e:
            print(f"Error during prediction: {e}")
            predicted_character = "Unknown"
            confidence = 0.0
            

        # Draw bounding box and predicted character on frame
        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10
        x2 = int(max(x_) * W) + 10
        y2 = int(max(y_) * H) + 10

        # Draw rectangle around detected hand and put text with prediction and confidence
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
        cv2.putText(frame, f'{predicted_character} - {confidence:.2f}%', 
                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

    # Show the frame with the prediction and confidence
    cv2.imshow('Sign Language Recognition', frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

#.\my_env\Scripts\activate  
#Set-ExecutionPolicy RemoteSigned
