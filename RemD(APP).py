import cv2
import mediapipe as mp
import numpy as np
import os
import math
import time

# Initialize MediaPipe for hand tracking
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Initialize webcam input
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 30)  # Set frames per second
width = 1280
height = 720
cap.set(3, width)  # Set width
cap.set(4, height)  # Set height

# Initialize canvas to draw on
imgCanvas = np.zeros((height, width, 3), np.uint8)

# Set initial drawing parameters
drawColor = (0, 0, 255)  # Red as initial color
previousColor = drawColor  # Store the previous color
thickness = 10  # Initial drawing thickness
tipIds = [4, 8, 12, 16, 20]  # Fingertip landmarks
xp, yp = [0, 0]  # Previous drawing coordinates

# Define a dictionary to map keys to colors
key_color_mapping = {
    'r': (0, 0, 255),  # Red (BGR)
    'b': (255, 0, 0),  # Blue (BGR)
    'g': (0, 255, 0),  # Green (BGR)
    'c': (255, 255, 0),  # Cyan (BGR)
    'm': (255, 0, 255),  # Magenta (BGR)
    'y': (0, 255, 255),  # Yellow (BGR)
    'k': (0, 0, 0)  # Black (eraser) (BGR)
}

# Define key bindings for thickness changes
thickness_mapping = {
    '1': 5,   # Pressing '1' sets thickness to 5
    '2': 10,  # Pressing '2' sets thickness to 10
    '3': 15,  # Pressing '3' sets thickness to 15
    '4': 20,  # Pressing '4' sets thickness to 20
    '5': 25,  # Pressing '5' sets thickness to 25
}

# Variables for the timer
two_fingers_up_start_time = None  # Track the start time of two fingers being up

# Define the toggle view variable
toggle_view = True  # Initially start with webcam view

# Initialize MediaPipe Hands
with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5, max_num_hands=1) as hands:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            break

        # Flip and convert image
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Process the image with MediaPipe Hands
        results = hands.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Initialize new hand detection
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                points = [(int(lm.x * width), int(lm.y * height)) for lm in hand_landmarks.landmark]
                x1, y1 = points[8]  # Index finger tip
                x2, y2 = points[12]  # Middle finger tip

                # Check which fingers are up
                fingers = []
                if points[4][0] < points[3][0]:
                    fingers.append(1)
                else:
                    fingers.append(0)

                for i in range(1, 5):
                    if points[tipIds[i]][1] < points[tipIds[i] - 2][1]:
                        fingers.append(1)
                    else:
                        fingers.append(0)

                # Check if the index and middle fingers are up
                index_and_middle_up = fingers[1] and fingers[2]

                # If the index and middle fingers are up
                if index_and_middle_up:
                    # Check if the timer is not started yet
                    if two_fingers_up_start_time is None:
                        # Start the timer
                        two_fingers_up_start_time = time.time()

                    # Calculate the time the two fingers have been up
                    time_elapsed = time.time() - two_fingers_up_start_time

                    # If the fingers have been up for more than 1 second
                    if time_elapsed >= 0.75:
                        # Toggle between eraser mode and the previous color
                        if drawColor == key_color_mapping['k']:
                            drawColor = previousColor
                        else:
                            previousColor = drawColor
                            drawColor = key_color_mapping['k']
                        
                        # Reset the timer
                        two_fingers_up_start_time = None
                else:
                    # Reset the timer if the fingers are not up
                    two_fingers_up_start_time = None

                # Calculate the distance between thumb tip and index finger tip
                thumb_tip = points[4]  # Thumb tip (landmark 4)
                index_finger_tip = points[8]  # Index finger tip (landmark 8)
                distance = math.sqrt((thumb_tip[0] - index_finger_tip[0]) ** 2 + (thumb_tip[1] - index_finger_tip[1]) ** 2)

                # Define a threshold for pinch gesture detection
                pinch_threshold = 50  # Adjust the threshold as needed

                # Check if the distance is below the threshold (pinch gesture detected)
                if distance < pinch_threshold:
                    print("Pinch gesture detected: Clearing the screen")
                    imgCanvas = np.zeros((height, width, 3), np.uint8)  # Clear the canvas

                # Draw mode - index finger is up and no other finger is up
                elif fingers[1] and not any(fingers[i] for i in range(2, 5)):
                    # Draw a filled circle at the index finger tip using the current color
                    cv2.circle(image, (x1, y1), thickness // 2, drawColor, cv2.FILLED)
                    
                    if xp == 0 and yp == 0:
                        xp, yp = x1, y1
                    
                    # Draw a line from the previous point to the current point
                    cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, thickness)
                    
                    # Update previous coordinates
                    xp, yp = x1, y1

        # Combine the canvas with the image
        imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
        _, imgInv = cv2.threshold(imgGray, 5, 255, cv2.THRESH_BINARY_INV)
        imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
        img = cv2.bitwise_and(image, imgInv)
        img = cv2.bitwise_or(img, imgCanvas)

        # Handle view toggling
        if toggle_view:
            # Combine the webcam view and the drawing board
            img = cv2.addWeighted(image, 0.5, imgCanvas, 0.5, 0)
        else:
            # Display only the drawing board
            img = imgCanvas
        
        # Listen for key presses and handle toggling
        key = cv2.waitKey(1) & 0xFF
        
        # Toggle the view if 't' key is pressed
        if key == ord('t'):
            toggle_view = not toggle_view  # Toggle between webcam and drawing board
        
        # Change the color based on key-color mapping
        if chr(key) in key_color_mapping:
            drawColor = key_color_mapping[chr(key)]
            print(f"Color changed to: {drawColor}")

        # Change the thickness based on thickness mapping
        if chr(key) in thickness_mapping:
            thickness = thickness_mapping[chr(key)]
            print(f"Thickness changed to: {thickness}")

        # Take a screenshot if 's' key is pressed
        if key == ord('s'):
            cv2.imwrite('drawing_screenshot.png', imgCanvas)
            print("Screenshot saved as drawing_screenshot.png")

        # Display the final image
        cv2.imshow('AI Virtual Painter', img)
        
        # Break the loop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the webcam and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
