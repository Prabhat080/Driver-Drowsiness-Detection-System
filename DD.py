from scipy.spatial import distance
from imutils import face_utils
import imutils
import dlib
import cv2

def eye_aspect_ratio(eye):
	#eye[] is a zero indexed array
	A = distance.euclidean(eye[1], eye[5]) #dist between pt 2 and pt 6
	B = distance.euclidean(eye[2], eye[4]) #dist between pt 3 and pt 5
	C = distance.euclidean(eye[0], eye[3]) #dist betweeen pt 1 and pt 4
	ear = (A + B) / (2.0 * C)
	return ear
	

frame_check = 15
cap=cv2.VideoCapture(0)

detect = dlib.get_frontal_face_detector() #returns a function to detect a face as an object
predict = dlib.shape_predictor(".\shape_predictor_68_face_landmarks.dat")# Dat file is the main part of the code

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]   #getting points on left eye
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]  #getting points on right eye

flag=0
while True:
	ret, frame=cap.read() #extracting image
	frame = imutils.resize(frame, width=1280,height=720 )
	frame=cv2.flip(frame,1)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  #converting to grayscale image
	subjects = detect(gray, 0)
	for subject in subjects:
		shape = predict(gray, subject)
		shape = face_utils.shape_to_np(shape) #converting to NumPy Array
		leftEye = shape[lStart:lEnd]
		rightEye = shape[rStart:rEnd]
		leftEAR = eye_aspect_ratio(leftEye)
		rightEAR = eye_aspect_ratio(rightEye)
		ear = (leftEAR + rightEAR) / 2.0          #avg EAR of both eyes 
		leftEyeHull = cv2.convexHull(leftEye)
		rightEyeHull = cv2.convexHull(rightEye)

		cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1) # drawing contours joining the points 
		cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1) #obtained on the eyes
		if ear < 0.25:
			flag += 1
			print ("Frame = ",flag,"   EAR = ",ear)
			if flag >= frame_check:  #frame_check=15
				cv2.putText(frame, "****************ALERT!****************", (400, 30),
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
				cv2.putText(frame, "****************ALERT!****************", (400,625),
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
				print ("Drowsy")
		else:
			flag = 0
	cv2.imshow("Frame", frame)        #continuous display of frames
	key = cv2.waitKey(1) & 0xFF
	if key == ord("q"):              #press q to stop the program
		break
cv2.destroyAllWindows()
cap.stop()