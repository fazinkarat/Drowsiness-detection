# Drowsiness-detection
CNN and TensorFlow is used to detect drowsiness by extracting features from images of people's eyes. The CNN model is trained on a dataset of images of drowsy and non-drowsy people. Once the model is trained, it can be used to detect drowsiness in real time by analyzing images of people's eyes

1. Import necessary libraries and modules:

   - `cv2`: OpenCV library for computer vision tasks.
   - `os`: Python library for interacting with the operating system.
   - `keras.models`: Keras, a high-level neural networks API running on top of TensorFlow.
   - `numpy`: NumPy for numerical operations.
   - `pygame.mixer`: Pygame library for sound mixing.
   - `time`: Python time module for timing-related functions.

2. Initialize Pygame mixer for playing an alarm sound and load the alarm sound from 'alarm.wav'.

3. Load Haar cascade classifiers for detecting faces, left eyes, and right eyes. These classifiers are used for detecting facial features in the video frames.

4. Define a list `lbl` with labels "Close" and "Open" to represent eye states.

5. Load a pre-trained convolutional neural network (CNN) model for drowsiness detection using Keras. The model is loaded from the 'cnncat2.h5' file.

6. Capture video from the default camera (0) using OpenCV (`cv2.VideoCapture`).

7. Define variables for font, counting frames, keeping score, and controlling the thickness of rectangles drawn on the video feed.

8. Start an infinite loop to continuously process video frames.

9. Inside the loop:

   - Read a frame from the video feed.
   - Convert the frame to grayscale.
   - Detect faces, left eyes, and right eyes using the Haar cascade classifiers.
   - Draw rectangles around detected faces on the frame.

10. For each detected right and left eye:

    - Preprocess the eye image by converting it to grayscale, resizing it to 24x24 pixels, and normalizing it.
    - Feed the preprocessed image into the loaded CNN model.
    - Predict whether the eye is "Open" or "Closed" based on the model's output.

11. Based on the predictions for both eyes, update the `score`:

    - If both eyes are closed, increment the `score`.
    - If at least one eye is open, decrement the `score`.

12. Display the current eye state and the score on the video frame.

13. If the `score` exceeds a threshold of 15, indicating that the person is feeling sleepy, the code takes the following actions:

    - Captures a still image from the video feed.
    - Plays an alarm sound using Pygame mixer.
    - Draws a red rectangle around the frame to alert the user.

14. The video frame with annotations is displayed in a window.

15. The loop continues until the user presses the 'q' key, at which point the video capture is released, and all OpenCV windows are destroyed.

This code combines face and eye detection with a pre-trained neural network to determine whether a person's eyes are open or closed, allowing it to detect drowsiness based on eye state. If you have any specific questions about how parts of this code work or need further explanations, feel free to ask!
