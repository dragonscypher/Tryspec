import cv2
import mediapipe as mp
import numpy as np
import os
import warnings
import absl.logging
import sys

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Paths to glasses and hats images
glasses_paths = [ "glasses3.png", "glasses4.png", "glasses6.png"]
hats_paths = ["hat2.png"]

# Function to load images with transparency
def load_images(paths):
    images = []
    for path in paths:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"Warning: Image at path {path} could not be loaded.")
        else:
            images.append(img)
    return images

# Load images for glasses and hats
glasses_images = load_images(glasses_paths)
hats_images = load_images(hats_paths)

# Current selection index
current_glasses_index = 0
current_hat_index = 0

# Function to overlay image at a given position
def overlay_image(background, overlay, position, size):
    if size[0] <= 0 or size[1] <= 0:
        print(f"Invalid size for overlay: {size}")
        return background
    
    x, y = position
    overlay_resized = cv2.resize(overlay, size, interpolation=cv2.INTER_AREA)
    overlay_height, overlay_width = overlay_resized.shape[:2]

    # Ensure position does not go out of bounds
    x = max(0, min(x, background.shape[1] - overlay_width))
    y = max(0, min(y, background.shape[0] - overlay_height))

    # Region of Interest (ROI)
    roi = background[y:y+overlay_height, x:x+overlay_width]

    # Check if overlay image has an alpha channel
    if overlay_resized.shape[2] == 4:
        # Extract alpha channel
        alpha = overlay_resized[:, :, 3] / 255.0
        overlay_rgb = overlay_resized[:, :, :3]

        # Blend overlay and background
        for c in range(0, 3):
            roi[:, :, c] = roi[:, :, c] * (1 - alpha) + overlay_rgb[:, :, c] * alpha
    else:
        # If no alpha channel, overlay directly (assuming it's already transparent)
        background[y:y+overlay_height, x:x+overlay_width] = overlay_resized

    return background

# Function to calculate rotation and scaling based on landmarks
def calculate_rotation_and_scaling(landmarks, w, h):
    left_eye = np.array([landmarks[33].x * w, landmarks[33].y * h])
    right_eye = np.array([landmarks[263].x * w, landmarks[263].y * h])
    nose_tip = np.array([landmarks[1].x * w, landmarks[1].y * h])

    # Calculate face width and height based on landmarks
    face_width = np.linalg.norm(right_eye - left_eye)
    face_height = np.linalg.norm(nose_tip - left_eye)

    # Estimate rotation based on the position of the eyes
    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]
    angle = np.degrees(np.arctan2(dy, dx))

    return face_width, face_height, angle

# Function to process each frame
def process_frame(frame, glasses_image, hat_image):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = mp_face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Convert normalized coordinates to pixel values
            h, w, _ = frame.shape

            # Calculate rotation, scaling, and landmark coordinates
            face_width, face_height, angle = calculate_rotation_and_scaling(face_landmarks.landmark, w, h)

            if face_width <= 0 or face_height <= 0:
                print(f"Invalid face dimensions: width={face_width}, height={face_height}")
                continue

            # Calculate glasses position and size dynamically based on eye coordinates
            glasses_width = int(1.7 * face_width)  # Increase width for better fitting
            glasses_height = int(1.0 * face_height)  # Further increase height for better fitting
            glasses_position = (int(face_landmarks.landmark[33].x * w) - glasses_width // 4, int(face_landmarks.landmark[168].y * h) - glasses_height // 3)

            # Overlay glasses
            frame = overlay_image(frame, glasses_image, glasses_position, (glasses_width, glasses_height))

            # Calculate hat position and size dynamically based on forehead coordinates
            hat_width = int(2.7 * face_width)  # Increase width for better fitting
            hat_height = int(2.1 * face_height)  # Further increase height for better fitting
            hat_position = (int(face_landmarks.landmark[10].x * w) - hat_width // 2, int(face_landmarks.landmark[10].y * h) - int(0.9 * hat_height))

            # Overlay hat
            frame = overlay_image(frame, hat_image, hat_position, (hat_width, hat_height))

    return frame

# Main function updated with selection and toggle functionality
def main():
    global current_glasses_index, current_hat_index

    cap = cv2.VideoCapture(0)  # 0 for the default camera

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Select current glasses and hat
        if current_glasses_index < len(glasses_images):
            glasses_image = glasses_images[current_glasses_index]
        else:
            print("No valid glasses image loaded.")
            glasses_image = None

        if current_hat_index < len(hats_images):
            hat_image = hats_images[current_hat_index]
        else:
            print("No valid hat image loaded.")
            hat_image = None

        if glasses_image is not None and hat_image is not None:
            processed_frame = process_frame(frame, glasses_image, hat_image)
            cv2.imshow("Live Feed", processed_frame)
        else:
            cv2.imshow("Live Feed", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):  # Press 'q' to exit
            break
        elif key == ord('g'):  # Press 'g' to toggle glasses
            current_glasses_index = (current_glasses_index + 1) % len(glasses_images)
        elif key == ord('h'):  # Press 'h' to toggle hats
            current_hat_index = (current_hat_index + 1) % len(hats_images)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()