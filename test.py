import cv2
import dlib
import numpy as np

# Initialize dlib's face detector and landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(r"shape_predictor_68_face_landmarks.dat")  # Provide the path to the model

# Paths to glasses and hats images
glasses_paths = ["glasses1.png", "glasses2.png", "glasses3.png","glasses4.png","glasses5.png","glasses6.png","glasses7.png",]
hats_paths = ["hat1.png", "hat2.png", "hat3.png"]

# Function to load images with transparency
def load_images(paths):
    return [cv2.imread(path, cv2.IMREAD_UNCHANGED) for path in paths]

# Load images for glasses and hats
glasses_images = load_images(glasses_paths)
hats_images = load_images(hats_paths)

# Define offsets for each pair of glasses (x_offset, y_offset)
glasses_offsets = [
    (-5, 9),  # No offset for the first pair
    (-7, -20),  # Slight left and down for the second pair
    (-10, 5),  # Right and up for the third pair
    (-9, 9),
    (-10, -9),
    (-10, 10),
    (-10, -12),
    # Add more as needed for each pair of glasses
]

# Current selection index
current_glasses_index = 0
current_hat_index = 0

# Function to overlay image at a given position
def overlay_image(background, overlay, position, scale=1.0):
    overlay_height, overlay_width = overlay.shape[:2]
    x, y = position

    # Scaling the overlay
    overlay = cv2.resize(overlay, (int(overlay_width * scale), int(overlay_height * scale)), interpolation=cv2.INTER_AREA)

    # Ensure position does not go out of bounds
    x = max(0, min(x, background.shape[1] - overlay.shape[1]))
    y = max(0, min(y, background.shape[0] - overlay.shape[0]))

    # Region of Interest (ROI)
    roi = background[y:y+overlay.shape[0], x:x+overlay.shape[1]]

    # Check if overlay image has an alpha channel
    if overlay.shape[2] == 4:
        # Extract alpha channel
        alpha = overlay[:, :, 3] / 255.0
        overlay_rgb = overlay[:, :, :3]

        # Blend overlay and background
        for c in range(0, 3):
            roi[:, :, c] = roi[:, :, c] * (1 - alpha) + overlay_rgb[:, :, c] * alpha
    else:
        # If no alpha channel, overlay directly (assuming it's already transparent)
        background[y:y+overlay.shape[0], x:x+overlay.shape[1]] = overlay

    return background

def calculate_tilt(landmarks):
    # Points for calculating tilt: right eye (45) and chin (8)
    x1, y1 = landmarks.part(36).x, landmarks.part(36).y  # Left eye
    x2, y2 = landmarks.part(45).x, landmarks.part(45).y  # Right eye

    # Calculating angle in degrees
    angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
    return angle

def rotate_image(image, angle):
    # Split alpha channel if present
    if image.shape[2] == 4:
        alpha_channel = image[:, :, 3]
        image_rgb = image[:, :, :3]
        
        # Rotate RGB image and alpha channel separately
        rotated_image_rgb = rotate_image_inner(image_rgb, angle)
        rotated_alpha = rotate_image_inner(alpha_channel, angle)

        # Merge them back
        rotated_image = np.dstack((rotated_image_rgb, rotated_alpha))
        return rotated_image

    # If no alpha channel, just rotate
    return rotate_image_inner(image, angle)

def rotate_image_inner(image, angle):
    height, width = image.shape[:2]
    center = (width / 2, height / 2)  # Center of the image

    # Rotation matrix
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Perform rotation
    result = cv2.warpAffine(image, rot_mat, (width, height))
    return result

def is_side_view(landmarks):
    # Compare the distances between the leftmost and rightmost points of the face
    # and the nose tip to see if the face is turned significantly.
    nose_tip = landmarks.part(30)  # Nose tip landmark
    leftmost = landmarks.part(0)   # Leftmost point of the face
    rightmost = landmarks.part(16) # Rightmost point of the face

    distance_left = np.linalg.norm(np.array([nose_tip.x, nose_tip.y]) - np.array([leftmost.x, leftmost.y]))
    distance_right = np.linalg.norm(np.array([nose_tip.x, nose_tip.y]) - np.array([rightmost.x, rightmost.y]))

    # If one side is significantly closer to the nose tip than the other, the face is likely turned.
    return abs(distance_left - distance_right) > 120  # Adjust the threshold as needed


def draw_ear_to_glasses_line(frame, landmarks):
    # Draw a line from the top of the ear to the glasses
    # You'll need to adjust the start and end points according to your requirements
    start_point = (landmarks.part(0).x, landmarks.part(0).y)  # Example start point
    end_point = (landmarks.part(36).x, landmarks.part(36).y)  # Example end point
    frame = cv2.line(frame, start_point, end_point, (0, 0, 250), thickness=2)
    return frame


# Function to process each frame
def process_frame(frame, glasses_image, hat_image):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)
        # Check if the face is in a side view
        if is_side_view(landmarks):
            continue  # Skip overlaying images for this face

        # Calculate tilt angle
        tilt_angle = calculate_tilt(landmarks)

        # Introduce a threshold for tilt sensitivity
        tilt_threshold = 5  # Adjust this value based on sensitivity preference
        if abs(tilt_angle) < tilt_threshold:
            tilt_angle = 0  # Ignore minor tilts
        # Debug: Print the tilt angle
        # print("Tilt angle:", tilt_angle)
        # Invert the angle
        tilt_angle = -tilt_angle

        # Rotate glasses image based on tilt angle
        rotated_glasses = rotate_image(glasses_image, tilt_angle)
        # Glasses overlay
        face_left_x = landmarks.part(0).x
        face_right_x = landmarks.part(16).x
        eye_y = (landmarks.part(36).y + landmarks.part(45).y) // 2
        face_width = face_right_x - face_left_x
        glasses_scale_factor = face_width / rotated_glasses.shape[1]
        # Get the offset for the current pair of glasses
        glasses_offset_x, glasses_offset_y = glasses_offsets[current_glasses_index]
         # Glasses overlay
        glasses_position = (face_left_x + 20+ glasses_offset_x, eye_y - 45 + glasses_offset_y) # Adjust as needed
        frame = overlay_image(frame, rotated_glasses, glasses_position, glasses_scale_factor)
         # For debugging: Draw a line for tilt calculation
        # cv2.line(frame, (landmarks.part(17).x, landmarks.part(17).y), (landmarks.part(26).x, landmarks.part(26).y), (0, 255, 0), 2)


        # Draw a line for side view if necessary
        if is_side_view(landmarks):
            frame = draw_ear_to_glasses_line(frame, landmarks)

    # Hat overlay
    face_width = landmarks.part(16).x - landmarks.part(0).x
    # Assuming the forehead_y position might need to be dynamically adjusted
    # Using a higher landmark for the forehead, then adjusting upwards a bit to fit the hat above
    forehead_y_adjustment = 150  # Adjust based on the hat image
    forehead_y = max(landmarks.part(19).y - forehead_y_adjustment, 0)  # Ensure it's not off-screen
    # Calculate dynamic scale factor based on face width and desired hat width
    desired_hat_width = face_width * 1.2  # Example: hat width 20% larger than face width
    # Dynamic scale factor for the hat based on face width
    hat_scale_factor = desired_hat_width / hat_image.shape[1]

    # Preparing hat image by scaling
    hat_resized = cv2.resize(hat_image, None, fx=hat_scale_factor, fy=hat_scale_factor, interpolation=cv2.INTER_AREA)

    # Adjust the hat position; consider the hat's height
    hat_position_x = face_left_x + (face_width - hat_resized.shape[1]) // 2
    hat_position_y = forehead_y - hat_resized.shape[0] // 3  # Adjust this division factor as needed

    # Ensure the hat's position is fully within the frame
    hat_position_y = max(0, hat_position_y)
    hat_position_x = max(0, min(hat_position_x, frame.shape[1] - hat_resized.shape[1]))
    # Center the hat horizontally based on the scaled width
    hat_position_x = face_left_x + (face_width - hat_resized.shape[1]) // 2

    # Apply the hat overlay; ensure resized hat is used
    frame = overlay_image(frame, hat_resized, (hat_position_x, hat_position_y), 1.0)  # Scale is 1.0 since it's already resized

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
        glasses_image = glasses_images[current_glasses_index]
        hat_image = hats_images[current_hat_index]

        processed_frame = process_frame(frame, glasses_image, hat_image)
        cv2.imshow("Live Feed", processed_frame)

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
