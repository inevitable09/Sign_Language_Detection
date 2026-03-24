import time
import cv2
import threading
import pyttsx3
import customtkinter as ctk
from PIL import Image, ImageTk
from inference_classifier import predict_sign_language

# Configure appearance
ctk.set_appearance_mode("Dark") # Can be "Light", "Dark", or "System"
ctk.set_default_color_theme("blue") # Can be blue, green, dark-blue etc.

class SignLanguageApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Sign Language Recognition")
        self.geometry("1200x800")
        self.resizable(False, False)
        
        # Sentence and logic trackers
        self.current_char = ""
        self.sentence = ""
        self.last_appended_char = ""

        # New variables for sign stability and warm-up
        self.last_predicted_sign = ""
        self.consecutive_frames_count = 0
        self.frame_threshold = 10
        self.warm_up_delay = 1.0
        self.hand_detected_time = None
        self.is_warmed_up = False
        
        # Layout
        self.setup_ui()
        
        # Webcam setup
        self.cap = cv2.VideoCapture(0)
        self.running = True
        self.update_video_frame()
        
        # Text-to-Speech engine and flag
        self.is_speaking = False

    def setup_ui(self):
        # Configure grid weights for responsive layout
        # Three columns: Chart1 | Video Feed | Chart2
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=2)
        self.grid_columnconfigure(2, weight=1)
        
        # Five rows:
        # Row 0: Chart1, Video, Chart2
        # Row 1: Character Label
        # Row 2: Sentence Box
        # Row 3: Chart 3
        # Row 4: Buttons
        self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=0)
        self.grid_rowconfigure(2, weight=0)
        self.grid_rowconfigure(3, weight=0)
        self.grid_rowconfigure(4, weight=0)

        # --- Chart 1 (A-L) ---
        try:
            self.chart1_image = Image.open("SignChart1.png")
            self.chart1_image = self.chart1_image.resize((300, 400), Image.LANCZOS)
            self.chart1_photo = ctk.CTkImage(light_image=self.chart1_image, dark_image=self.chart1_image, size=(300, 400))
            self.chart1_label = ctk.CTkLabel(self, image=self.chart1_photo, text="")
            self.chart1_label.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        except Exception as e:
            print(f"Could not load sign chart 1 image: {e}")
            self.chart1_label = ctk.CTkLabel(self, text="Error loading chart 1.")
            self.chart1_label.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        # --- Webcam Feed ---
        self.video_label = ctk.CTkLabel(self, text="")
        self.video_label.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")

        # --- Chart 2 (M-X) ---
        try:
            self.chart2_image = Image.open("SignChart2.png")
            self.chart2_image = self.chart2_image.resize((300, 400), Image.LANCZOS)
            self.chart2_photo = ctk.CTkImage(light_image=self.chart2_image, dark_image=self.chart2_image, size=(300, 400))
            self.chart2_label = ctk.CTkLabel(self, image=self.chart2_photo, text="")
            self.chart2_label.grid(row=0, column=2, padx=10, pady=10, sticky="nsew")
        except Exception as e:
            print(f"Could not load sign chart 2 image: {e}")
            self.chart2_label = ctk.CTkLabel(self, text="Error loading chart 2.")
            self.chart2_label.grid(row=0, column=2, padx=10, pady=10, sticky="nsew")

        # # --- Predicted Character Display ---
        # self.char_label = ctk.CTkLabel(self, text="Character: ", font=ctk.CTkFont(size=20, weight="bold"))
        # # Character label is now centered under the video feed
        # self.char_label.grid(row=1, column=1, pady=(10, 5), sticky="ew")

        # # --- Sentence Display ---
        # self.sentence_box = ctk.CTkTextbox(self, height=80, font=ctk.CTkFont(size=16))
        # # Sentence box is now centered under the video feed
        # self.sentence_box.grid(row=2, column=1, pady=(5, 10), padx=20, sticky="ew")
        # self.sentence_box.insert("0.0", "")
        # self.sentence_box.configure(state="disabled")

        # --- Chart 3 (Y, Z, Del, Space) ---
        try:
            self.chart3_image = Image.open("SignChart3.png")
            self.chart3_image = self.chart3_image.resize((600, 100), Image.LANCZOS)
            self.chart3_photo = ctk.CTkImage(light_image=self.chart3_image, dark_image=self.chart3_image, size=(600, 100))
            self.chart3_label = ctk.CTkLabel(self, image=self.chart3_photo, text="")
            # Chart 3 is now placed directly in the video column
            self.chart3_label.grid(row=1, column=1, padx=20, pady=10, sticky="nsew")
        except Exception as e:
            print(f"Could not load sign chart 3 image: {e}")
            self.chart3_label = ctk.CTkLabel(self, text="Error loading chart 3.")
            self.chart3_label.grid(row=1, column=1, padx=20, pady=10, sticky="nsew")

        # --- Predicted Character Display ---
        self.char_label = ctk.CTkLabel(self, text="Character: ", font=ctk.CTkFont(size=20, weight="bold"))
        # Character label is now centered under the video feed
        self.char_label.grid(row=2, column=1, pady=(10, 5), sticky="ew")

        # --- Sentence Display ---
        self.sentence_box = ctk.CTkTextbox(self, height=80, font=ctk.CTkFont(size=16))
        # Sentence box is now centered under the video feed
        self.sentence_box.grid(row=3, column=1, pady=(5, 10), padx=20, sticky="ew")
        self.sentence_box.insert("0.0", "")
        self.sentence_box.configure(state="disabled")

        # --- Buttons Frame ---
        button_frame = ctk.CTkFrame(self)
        button_frame.grid(row=4, column=1, pady=10)

        self.clear_btn = ctk.CTkButton(button_frame, text="Clear", command=self.clear_sentence, width=120)
        self.clear_btn.pack(side="left", padx=10)

        self.speak_btn = ctk.CTkButton(button_frame, text="Speak", command=self.speak_sentence, width=120)
        self.speak_btn.pack(side="left", padx=10)

    # ... (rest of your class methods)
    def update_video_frame(self):
        if not self.running:
            return

        ret, frame = self.cap.read()
        if ret:
            frame_for_ui = cv2.flip(frame, 1) # Mirror image for better user experience
            rgb_frame = cv2.cvtColor(frame_for_ui, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb_frame)
            imgtk = ctk.CTkImage(light_image=img, size=(640, 480)) # Adjusted size for video feed

            self.video_label.configure(image=imgtk)
            self.video_label.image = imgtk

            # Predict sign language character
            predicted_char, _ = predict_sign_language(frame)
            self.char_label.configure(text=f"Character: {predicted_char}")

            # --- NEW LOGIC: Warm-up, Hold-to-Capture, and Separator ---
            is_valid_sign = predicted_char not in ["Unknown", "No hands detected"]

            # 1. Warm-up Period: Wait for a few seconds after the first hand is detected
            if is_valid_sign and not self.hand_detected_time:
                self.hand_detected_time = time.time()
            
            if self.hand_detected_time and (time.time() - self.hand_detected_time) >= self.warm_up_delay:
                self.is_warmed_up = True
            
            # 2. Hold-to-Capture and Separator: Only process if the app has warmed up
            if self.is_warmed_up:
                # Check for a separator or neutral sign to reset the logic
                # You must train your model to recognize a "Neutral" sign.
                if predicted_char == "Neutral":
                    self.last_predicted_sign = ""
                    self.consecutive_frames_count = 0
                    # Do NOT update last_appended_char here, otherwise it will prevent
                    # the next identical character from being appended.
                    self.last_appended_char = "Neutral_Reset" # Use a special temporary value
                    self.char_label.configure(text="Character: Ready for next sign...")
                    # No return here, let the loop continue to ensure UI refreshes correctly
                else: # Only proceed with normal sign detection if it's not a Neutral sign
                    if predicted_char == self.last_predicted_sign and is_valid_sign:
                        self.consecutive_frames_count += 1
                    else:
                        self.last_predicted_sign = predicted_char
                        self.consecutive_frames_count = 0
                    
                    # If a sign has been stable for a threshold of frames, append it
                    if self.consecutive_frames_count >= self.frame_threshold:
                        # Append the character if it's not a special command and different from the last
                        # or if the last appended char was the special reset value
                        if self.last_predicted_sign != self.last_appended_char or self.last_appended_char == "Neutral_Reset":
                            # Handle "Space" or "Del" signs
                            if self.last_predicted_sign == "Space":
                                if self.sentence and self.sentence[-1] != ' ':
                                    self.sentence += " "
                            elif self.last_predicted_sign == "Del":
                                if self.sentence:
                                    self.sentence = self.sentence[:-1]
                            elif self.last_predicted_sign == "Neutral":
                                pass # Do nothing if Neutral is detected here, it was handled above
                            else:
                                self.sentence += self.last_predicted_sign
                            
                            self.update_sentence_box()
                            self.last_appended_char = self.last_predicted_sign
                            self.consecutive_frames_count = 0 # Reset counter after appending
            
        self.after(10, self.update_video_frame)

    def get_prediction(self, frame):
        predicted_character, _ = predict_sign_language(frame)
        return predicted_character

    def update_sentence_box(self):
        self.sentence_box.configure(state="normal")
        self.sentence_box.delete("0.0", "end")
        self.sentence_box.insert("0.0", self.sentence)
        self.sentence_box.configure(state="disabled")

    def clear_sentence(self):
        self.sentence = ""
        self.update_sentence_box()
        # Reset the hold-to-capture logic
        self.last_appended_char = ""
        self.last_predicted_sign = ""
        self.consecutive_frames_count = 0
        self.hand_detected_time = None
        self.is_warmed_up = False

    def speak_sentence(self):
        text = self.sentence.strip()
        print(f"Speaking: {text}")
        if text and not self.is_speaking:
            self.is_speaking = True
            self.speak_btn.configure(state="disabled")

            def _speak():
                try:
                    engine = pyttsx3.init()
                    engine.say(text)
                    engine.runAndWait()
                except Exception as e:
                    print(f"Error during speech: {e}")
                finally:
                    self.after(0, self._reset_speak_button)

            threading.Thread(target=_speak, daemon=True).start()
        else:
            print("No sentence to speak or already speaking.")

    def _reset_speak_button(self):
        self.is_speaking = False
        self.speak_btn.configure(state="normal")

    def on_closing(self):
        self.running = False
        if self.cap.isOpened():
            self.cap.release()
        self.destroy()

if __name__ == "__main__":
    app = SignLanguageApp()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()
