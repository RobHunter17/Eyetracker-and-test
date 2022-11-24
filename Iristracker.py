import cv2 as cv
import numpy as np
import mediapipe as mp
import math
import dlib
mp_face_mesh = mp.solutions.face_mesh

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


#looks for the first available video source on the device(desktop, laptop)
cap = cv.VideoCapture(0)

#references face map positioning of iris and eye corners
LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]
L_H_LEFT = [33]    
L_H_RIGHT = [133]   
R_H_LEFT = [362]    
R_H_RIGHT = [263]   

#takes the distance from one side of the eye to the other and returns the value
def euclidean_distance(point1, point2):
    x1, y1 =point1.ravel()
    x2, y2 =point2.ravel()
    distance = math.sqrt((x2-x1)**2 + (y2-y1)**2)
    return distance

#determines whether the iris' position is left right or center 
def iris_position(iris_center, right_point, left_point):
    center_to_right_dist = euclidean_distance(iris_center, right_point)
    total_distance = euclidean_distance(right_point, left_point)
    ratio = center_to_right_dist/total_distance
    iris_position = ""
    if ratio <= 2.5:
        iris_position = "left"
    elif ratio > 2.51 and ratio <= 2.92:
        iris_position = "center"
    else:
        iris_position = "right"
    return iris_position, ratio

#references the face mesh and then creates the loop for the pop up window to display the video and detection features
with mp_face_mesh.FaceMesh(max_num_faces = 1, refine_landmarks = True, min_detection_confidence = 0.5, min_tracking_confidence = 0.5) as face_mesh:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv.flip(frame, 1)
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        gray_frame = cv.cvtColor(rgb_frame, cv.COLOR_BGR2GRAY)
        gray_frame = cv.GaussianBlur(gray_frame, (9, 9), 0)
        _, threshold = cv.threshold(gray_frame, 45, 255, cv.THRESH_BINARY_INV)
        _, contours = cv.findContours(threshold, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        #print(contours)
        img_h, img_w = frame.shape[:2]
        results = face_mesh.process(rgb_frame)

        faces = detector(gray_frame)
        for face in faces:
            landmarks = predictor(gray_frame, face)
            left_eye_frame = np.array([(landmarks.part(36).x, landmarks.part(36).y),
									    (landmarks.part(37).x, landmarks.part(37).y),
									    (landmarks.part(38).x, landmarks.part(38).y),
									    (landmarks.part(39).x, landmarks.part(39).y),
									    (landmarks.part(40).x, landmarks.part(40).y),
									    (landmarks.part(41).x, landmarks.part(41).y)], np.int32)

            right_eye_frame = np.array([(landmarks.part(42).x, landmarks.part(42).y),
									    (landmarks.part(43).x, landmarks.part(43).y),
									    (landmarks.part(44).x, landmarks.part(44).y),
									    (landmarks.part(45).x, landmarks.part(45).y),
									    (landmarks.part(46).x, landmarks.part(46).y),
									    (landmarks.part(47).x, landmarks.part(47).y)], np.int32)

        min_x = np.min(left_eye_frame[:, 0])
        max_x = np.max(left_eye_frame[:, 0])
        min_y = np.min(left_eye_frame[:, 1])
        max_y = np.max(left_eye_frame[:, 1])

        min_x1 = np.min(right_eye_frame[:, 0])
        max_x1 = np.max(right_eye_frame[:, 0])
        min_y1 = np.min(right_eye_frame[:, 1])
        max_y1 = np.max(right_eye_frame[:, 1])

        if results.multi_face_landmarks:
            mesh_points = np.array([np.multiply([p.x, p.y], [img_w, img_h]).astype(int) for p in results.multi_face_landmarks[0].landmark])

            
            (l_cx, l_cy), l_radius = cv.minEnclosingCircle(mesh_points[LEFT_IRIS])
            (r_cx,r_cy), r_radius = cv.minEnclosingCircle(mesh_points[RIGHT_IRIS])

            center_left = np.array([l_cx, l_cy], dtype=np.int32)
            center_right = np.array([r_cx, r_cy], dtype=np.int32)

            height, width, _ = frame.shape
            mask = np.zeros((height, width), np.uint8)
            cv.polylines(mask, [left_eye_frame], True, 255, 2)
            cv.fillPoly(mask, [left_eye_frame], 255)
            left_eye = cv.bitwise_and(gray_frame, gray_frame, mask = mask)

            height, width, _ = frame.shape
            mask = np.zeros((height, width), np.uint8)
            cv.polylines(mask, [right_eye_frame], True, 255, 2)
            cv.fillPoly(mask, [right_eye_frame], 255)
            right_eye = cv.bitwise_and(gray_frame, gray_frame, mask = mask)

            gray_eye = left_eye[min_y: max_y, min_x: max_x]
            _, threshold_eye = cv.threshold(gray_eye, 75, 255, cv.THRESH_BINARY)
            height, width = threshold_eye.shape
            gray_eye_2 = right_eye[min_y1: max_y1, min_x1: max_x1]
            _, threshold_eye_2 = cv.threshold(gray_eye_2, 75, 255, cv.THRESH_BINARY)
            height, width = threshold_eye_2.shape

            threshold_eye = cv.resize(threshold_eye, None, fx = 5, fy = 5)
            lefteye = cv.resize(gray_eye, None, fx = 5, fy = 5)
            threshold_eye_right = cv.resize(threshold_eye_2, None, fx = 5, fy = 5)
            righteye = cv.resize(gray_eye_2, None, fx = 5, fy = 5)

            left_side_thresh = threshold_eye[0: height, 0: int(width / 2)]
            left_eye_left_side_pos = cv.countNonZero(left_side_thresh)
            left_eye_right_side_thresh = threshold_eye[0: height, int(width / 2): width]
            left_eye_right_side_pos = cv.countNonZero(left_eye_right_side_thresh)

            right_side_thresh = threshold_eye_2[0: height, 0: int(width / 2)]
            right_eye_left_side_pos = cv.countNonZero(right_side_thresh)
            right_eye_right_side_thresh = threshold_eye_2[0: height, int(width / 2): width]
            right_eye_right_side_pos = cv.countNonZero(right_eye_right_side_thresh)

            cv.putText(frame, str(left_eye_left_side_pos), (50, 100), cv.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)
            cv.putText(frame, str(left_eye_right_side_pos), (50, 125), cv.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)
            cv.putText(frame, str(right_eye_left_side_pos), (150, 100), cv.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)
            cv.putText(frame, str(right_eye_right_side_pos), (150, 125), cv.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)

            #Draws circles around the iris position
            cv.circle(frame, center_left, int(l_radius), (0, 255, 0), 1, cv.LINE_AA)
            cv.circle(frame, center_right, int(r_radius), (0, 255, 0), 1, cv.LINE_AA)

            #creates the values associated with the iris position and then displays that text on the screen
            iris_pos, ratio = iris_position(center_right, mesh_points[R_H_RIGHT], mesh_points[R_H_LEFT][0])
            cv.putText(frame, f"Iris Position: {iris_pos} {ratio: .2f}", (30,30), cv.FONT_HERSHEY_TRIPLEX, 1.2, (0,255,0), 1, cv.LINE_AA)

        _, eye_threshold = cv.threshold(lefteye, 50, 255, cv.THRESH_BINARY_INV)
        _, eye_contours = cv.findContours(eye_threshold, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        _, eye_threshold_right = cv.threshold(righteye, 50, 255, cv.THRESH_BINARY_INV)
        _, eye_contours_right = cv.findContours(eye_threshold_right, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        
        
        
        
        #print(eye_contours)
        #creates the display with video source
        cv.imshow("Iris Tracker", frame)
        #cv.imshow('Left eye Threshold', eye_threshold)
        #cv.imshow('Right eye threshold', eye_threshold_right)
        #cv.imshow('Left eye', lefteye)
        #cv.imshow('Right eye', righteye)
        key = cv.waitKey(1)
        if key == ord("q") or key == ord("Q"):
            break
cap.release()
cv.destroyAllWindows()