import cv2
import mediapipe as mp

# Initialize MediaPipe Hand tracking
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Open the webcam
cap = cv2.VideoCapture(0)

# Define button coordinates (top-left and bottom-right corners)
button_top_left = (10, 400)
button_bottom_right = (150, 450)
button_color = (0, 0, 255)  # Red
button_text = "EXIT"
exit_requested = False  # Global flag

# Finger counting logic
def count_fingers(hand_landmarks):
    finger_tips = [4, 8, 12, 16, 20]
    fingers_up = [0] * 5
    for i, tip in enumerate(finger_tips):
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y:
            fingers_up[i] = 1
    total_fingers = sum(fingers_up)
    AUTHORIZED_GESTURE = [1, 1, 0, 0, 1]  # Thumb, Index, Pinky

    if fingers_up == AUTHORIZED_GESTURE:
        return total_fingers, "Authenticated  ✅"
    else:
        return total_fingers, "Access Denied ❌"


# Mouse click handler
def mouse_callback(event, x, y, flags, param):
    global exit_requested
    if event == cv2.EVENT_LBUTTONDOWN:
        if button_top_left[0] <= x <= button_bottom_right[0] and button_top_left[1] <= y <= button_bottom_right[1]:
            exit_requested = True

cv2.namedWindow("Gesture Detection")
cv2.setMouseCallback("Gesture Detection", mouse_callback)

# Main loop
try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                fingers_up, gesture = count_fingers(hand_landmarks)

                cv2.putText(frame, f"Fingers: {fingers_up}", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                if gesture:
                    cv2.putText(frame, gesture, (50, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Draw exit button
        cv2.rectangle(frame, button_top_left, button_bottom_right, button_color, -1)
        cv2.putText(frame, button_text, (button_top_left[0] + 20, button_top_left[1] + 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.imshow("Gesture Detection", frame)

        key = cv2.waitKey(10)
        if key == ord('q') or key == 27 or exit_requested:
            break

except KeyboardInterrupt:
    print("Interrupted by user")

finally:
    cap.release()
    cv2.destroyAllWindows()