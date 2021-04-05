# 0. Install and import Dependecies
# !pip install mediapipe opencv_python

import mediapipe as mp
import cv2

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

# 1. Get realtime Webcam Feed
# cap = cv2.VideoCapture(0)
# while cap.isOpened():
#   ret, frame = cap.read()
#   cv2.imshow('Raw Webcam Feed', frame)
#
#   if cv2.waitKey(10) & 0xFF == ord('q'):
#     break
#
# cap.release()
# cv2.destroyAllWindows()
#
# cap.release()
# cv2.destroyAllWindows()


# 2. Make Detections from Feed
# cap = cv2.VideoCapture(0)
# # Initiate holistic model
# with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
#
#   while cap.isOpened():
#     ret, frame = cap.read()
#
#     # Recolor Feed
#     image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     # Make Detection
#     results = holistic.process(image)
#     print(results.pose_landmarks)
#
#     # Recolor image back to BGR for rendering
#     image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
#
#     # Draw face landmarks
#     mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACE_CONNECTIONS)
#
#     # Right hand
#     mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
#
#     # Left hand
#     mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
#
#     # Pose Detections
#     mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
#
#     cv2.imshow('Raw Webcam Feed', image)
#
#     if cv2.waitKey(10) & 0xFF == ord('q'):
#       break
#
# cap.release()
# cv2.destroyAllWindows()


# 3. Applying Styling

cap = cv2.VideoCapture(0)
# Initiate holistic model
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:

  while cap.isOpened():
    ret, frame = cap.read()

    # Recolor Feed
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Make Detection
    results = holistic.process(image)
    print(results.pose_landmarks)

    # Recolor image back to BGR for rendering
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Draw face landmarks
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACE_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(255,0,0), thickness=1, circle_radius=1),
                              mp_drawing.DrawingSpec(color=(0,255,0), thickness=1, circle_radius=1))

    # Right hand
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(255,0,0), thickness=2, circle_radius=2),
                              mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2))

    # Left hand
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(255,0,0), thickness=2, circle_radius=2),
                              mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2))

    # Pose Detections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(255,0,0), thickness=2, circle_radius=2),
                              mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2))

    cv2.imshow('Raw Webcam Feed', image)

    if cv2.waitKey(10) & 0xFF == ord('q'):
      break

cap.release()
cv2.destroyAllWindows()