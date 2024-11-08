import cv2
import mediapipe as mp
import numpy as np
import time
# from playsound import playsound
import os
import time
import threading

# Index Corresponding to Angle Value
ANGLES_DICT = {"hip_angle": 0,
               "knee_angle": 1,
               "ankle_angle": 2}

GLOBAL_SQUAT_COUNTER = 0
GLOBAL_INCORRECT_SQUAT_COUNTER = 0

delay = 10000
last_time = 0

current_state = 0
squat_performed = False

history = []


# def play_sound_in_background(sound_file):
#     # Create a thread that plays the sound
#     sound_thread = threading.Thread(target=playsound, args=(sound_file,))
#     sound_thread.start()  # Start the thread to play sound in the background


# TODO: Finish calculating angle function
def calculate_angle(point1, point2, point3):
    # Convert points to vectors
    vector1 = np.array([point1[0] - point2[0], point1[1] - point2[1]])
    vector2 = np.array([point3[0] - point2[0], point3[1] - point2[1]])

    dot_product = np.dot(vector1, vector2)
    magnitude1 = np.linalg.norm(vector1)
    magnitude2 = np.linalg.norm(vector2)

    if magnitude1 == 0 or magnitude2 == 0:
        print("Warning: One of the vectors has zero magnitude.")
        return 0.0

    angle_rad = np.arccos(dot_product / (magnitude1 * magnitude2))
    angle_deg = np.degrees(angle_rad)

    if np.isnan(angle_deg):
        return 0.0

    return angle_deg


'''
Heuristic Idea for Squat States

Goal is for the user to cycle from State 1 to State 3 back to State 1 Sequentially

If they go through 1 -> 3 -> 1 in proper form with no bad angles
    Increment Correct form counter

else if they go through 1 -> 2 and back down to 1 or detect any bad angles
    Increment Wrong form counter

We should save the state of the wrong form images and landmarks so that we can use llm for example to
explain how the user can improve their form.

 State 1: Initial Upright State
 State 2: Half Squat
 State 3: Full Squat

'''


def state_heuristic(new_state):
    global squat_performed
    global current_state
    # if(current_state == 0 and new_state == 1):
    #     current_state = new_state
    # elif(current_state == 1 and new_state == 2):
    #     current_state = new_state
    # elif(current_state == 2 and new_state == 3):
    #     current_state = new_state
    #     squat_performed = True
    #     return True, True
    # elif(current_state == 2 and new_state == 1 and not squat_performed):
    #     current_state = new_state
    #     squat_performed = True
    #     return False, True

    if (current_state == 2 and new_state == 3):
        current_state = new_state
        squat_performed = True
        return True, True
    # elif(current_state == 2 and new_state == 1 and squat_performed):

    elif (current_state == 2 and new_state == 1 and not squat_performed):
        current_state = new_state
        squat_performed = True
        return False, True

    current_state = new_state

    return False, False


# TODO: Implement
def determine_state(angles):
    global current_state
    # State 1
    if (angles[ANGLES_DICT["hip_angle"]] > 150 and angles[ANGLES_DICT["knee_angle"]] > 150 and angles[
        ANGLES_DICT["ankle_angle"]] > 100):
        return 1

    # State 2
    if (angles[ANGLES_DICT["hip_angle"]] > 90 and angles[ANGLES_DICT["knee_angle"]] > 110 and angles[
        ANGLES_DICT["ankle_angle"]] > 90):
        return 2

    # State 3
    if (angles[ANGLES_DICT["hip_angle"]] > 50 and angles[ANGLES_DICT["knee_angle"]] > 40 and angles[
        ANGLES_DICT["ankle_angle"]] > 70):
        return 3

    # Weird Calc -> Return current state
    return 0


def is_landmark_in_screen(landmarks):
    # Check if x and y are between 0 and 1 (within screen boundaries)
    if 0 <= landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x <= 1 and 0 <= landmarks[
        mp_pose.PoseLandmark.LEFT_SHOULDER.value].y <= 1:
        if 0 <= landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x <= 1 and 0 <= landmarks[
            mp_pose.PoseLandmark.LEFT_HIP.value].y <= 1:
            if 0 <= landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x <= 1 and 0 <= landmarks[
                mp_pose.PoseLandmark.LEFT_KNEE.value].y <= 1:
                if 0 <= landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x <= 1 and 0 <= landmarks[
                    mp_pose.PoseLandmark.LEFT_ANKLE.value].y <= 1:
                    if 0 <= landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].x <= 1 and 0 <= landmarks[
                        mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].y <= 1:
                        return True
    return False


# TODO: Classifying pose function
def classifying_squat(landmarks, display=True):
    # Correct Squat performed or not
    # label = False
    # performed_squat = False

    # Color of Text Label
    color = (0, 0, 225)

    if (not is_landmark_in_screen(landmarks)):
        return False, False

        # Convert landmark positions from normalized to pixel coordinates

    left_shoulder = (int(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * frame.shape[1]),
                     int(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * frame.shape[0]))
    left_hip = (int(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x * frame.shape[1]),
                int(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y * frame.shape[0]))
    left_knee = (int(landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x * frame.shape[1]),
                 int(landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y * frame.shape[0]))
    left_ankle = (int(landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x * frame.shape[1]),
                  int(landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y * frame.shape[0]))
    left_foot_index = (int(landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].x * frame.shape[1]),
                       int(landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].y * frame.shape[0]))

    # print("Shoulder Coords:", int(left_shoulder*100)/100)
    # print("Hip coordinates:", int(left_hip*100)/100)
    # print("Knee coordinates:", int(left_knee*100)/100)
    # print("Ankle coordinates:", int(left_ankle*100)/100)
    # print("Left Foot Index coordinates:", int(left_foot_index*100)/100)

    left_hip_angle = int(calculate_angle(left_shoulder, left_hip, left_knee))
    left_knee_angle = int(calculate_angle(left_hip, left_knee, left_ankle))
    left_ankle_angle = int(calculate_angle(left_knee, left_ankle, left_foot_index))
    # print("Calculated Left Hip angle:", left_knee_angle)
    # print("Calculated Left Knee angle:", left_knee_angle)
    # print("Calculated Left Ankle angle:", left_knee_angle)

    if (left_hip_angle == 0 or left_knee_angle == 0 or left_ankle_angle == 0):
        return False, False

    if (display):
        # Display Left Hip Angle
        cv2.putText(frame, str(left_hip_angle),
                    left_hip,
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

        # DisplayLeft Knee Angle
        cv2.putText(frame, str(left_knee_angle),
                    left_knee,
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

        # Display Left Ankle Angle
        cv2.putText(frame, str(int(left_ankle_angle)),
                    left_ankle,
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

    angles = (left_hip_angle, left_knee_angle, left_ankle_angle)

    new_state = determine_state(angles)
    print(new_state)

    return state_heuristic(new_state)

    # return label, performed_squat


# Initialize MediaPipe Pose and webcam
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
cap = cv2.VideoCapture(0)

# Runs Live Stream
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)

    # Squat State checker
    # Should define states that should be hit whenever squat is performed

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

        landmarks = results.pose_landmarks.landmark
        current_time = time.time()
        if (current_time - last_time < delay):
            continue
        correct_squat, performed_squat = classifying_squat(landmarks)

        if (correct_squat and performed_squat and squat_performed):
            GLOBAL_SQUAT_COUNTER += 1
            # if os.path.exists("ding.mp3"):
            #     play_sound_in_background("ding.mp3")
        elif (not correct_squat and performed_squat and squat_performed):  # INCORRECT DOES NOT WORK
            GLOBAL_INCORRECT_SQUAT_COUNTER += 1

        squat_performed = False

        # Display Number of Correct Squats Performed
        cv2.putText(frame, str(f"CORRECT SQUATS: {GLOBAL_SQUAT_COUNTER}"),
                    (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 225, 0), 2, cv2.LINE_AA)

        # TODO: SHIT BROKEN NEED TO FIX INCORRECT SQUATS HEURISTIC
        # # Display Number of Incorrect Squats Performed
        # cv2.putText(frame, str(f"INCORRECT SQUATS: {GLOBAL_INCORRECT_SQUAT_COUNTER}"),
        #             (50, 100),
        #             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 225), 2, cv2.LINE_AA)

    # For Selfie view
    # cv2.imshow('MediaPipe Pose', cv2.flip(frame, 1))

    cv2.imshow('Squat Form Corrector', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()