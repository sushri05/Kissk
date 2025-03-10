import cv2
import mediapipe as mp
import time

# Initialize MediaPipe Hands module
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Initialize webcam
cap = cv2.VideoCapture(0)

photo_count = 0  # Counter for saved images
gesture_start_time = None  # Timer variable
capture_scheduled = False  # Flag to indicate a scheduled capture

def is_custom_gesture(landmarks):
    """
    Detects if the hand is making the gesture in the uploaded image:
    - Index finger and thumb extended
    - Middle, ring, and pinky fingers curled
    """
    thumb_tip = landmarks[4]
    index_tip = landmarks[8]
    middle_tip = landmarks[12]
    ring_tip = landmarks[16]
    pinky_tip = landmarks[20]

    thumb_mcp = landmarks[2]
    index_mcp = landmarks[5]
    middle_mcp = landmarks[9]

    # Conditions for detecting the gesture
    is_thumb_extended = thumb_tip.y < thumb_mcp.y  # Thumb is pointing up
    is_index_extended = index_tip.y < index_mcp.y  # Index finger is up
    is_middle_curled = middle_tip.y > middle_mcp.y  # Middle finger curled
    is_ring_curled = ring_tip.y > middle_mcp.y  # Ring finger curled
    is_pinky_curled = pinky_tip.y > middle_mcp.y  # Pinky curled

    return (is_thumb_extended and is_index_extended and
            is_middle_curled and is_ring_curled and is_pinky_curled)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Convert to RGB for MediaPipe processing
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get landmark positions
            landmarks = hand_landmarks.landmark
            
            if is_custom_gesture(landmarks) and not capture_scheduled:
                gesture_start_time = time.time()  # Start the timer
                capture_scheduled = True  # Mark that capture is scheduled
                print("Gesture detected! Image will be captured in 5 seconds...")

    # If capture is scheduled, check the timer
    if capture_scheduled and time.time() - gesture_start_time >= 5:
        print("Capturing image...")
        photo_count += 1
        filename = f"captured_image_{photo_count}.jpg"
        cv2.imwrite(filename, frame)
        capture_scheduled = False  # Reset capture flag

    # Display output
    cv2.imshow("Gesture Capture", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Quit with 'q' key
        break

cap.release()
cv2.destroyAllWindows()
