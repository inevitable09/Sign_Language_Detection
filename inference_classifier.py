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

# Mediapipe hands module setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.3,
    min_tracking_confidence=0.3
)

# Labels dictionary (for A-Z, Del, Space)
labels_dict = {i: chr(65 + i) for i in range(26)}
labels_dict[26] = 'Space'
labels_dict[27] = 'Del'

def predict_sign_language(frame):
    data_aux = []
    x_ = []
    y_ = []

    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Extract x and y coordinates of landmarks
            for lm in hand_landmarks.landmark:
                x_.append(lm.x)
                y_.append(lm.y)

            for lm in hand_landmarks.landmark:
                data_aux.append(lm.x - min(x_))
                data_aux.append(lm.y - min(y_))

        # Predict using the trained model
        try:
            prediction_probabilities = model.predict_proba([np.asarray(data_aux)])
            predicted_class_index = np.argmax(prediction_probabilities)
            confidence = prediction_probabilities[0][predicted_class_index] * 100
            predicted_character = labels_dict[int(predicted_class_index)]

        except Exception as e:
            print(f"Error during prediction: {e}")
            return "Unknown", ""

        return predicted_character, predicted_character
    else:
        return "No hands detected", ""