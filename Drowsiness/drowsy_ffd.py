from imutils import face_utils
import dlib
import cv2
from pygame import mixer

threshold = 6

mixer.init()
sound = mixer.Sound('alarm.wav')

dlist = []

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
cap = cv2.VideoCapture(0)


def dist(a, b):
    x1, y1 = a
    x2, y2 = b
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5


while True:
    # Getting out image by webcam 
    _, image = cap.read()
    # Converting the image to gray scale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Get faces into webcam's image
    faces = detector(gray, 0)

    # For each detected face, find the landmark.
    for (i, face) in enumerate(faces):

        # Square around face
        cv2.rectangle(image, (face.left(),face.top()), (face.right(),face.bottom()), (255, 255, 0), 2)

        # Make the prediction and transform it to numpy array
        shape = predictor(gray, face)
        shape = face_utils.shape_to_np(shape)

        # Draw on our image, all the found coordinate points (x,y)
        for (x, y) in shape:
            cv2.circle(image, (x, y), 2, (0, 255, 0), -1)

        le_38 = shape[37]
        le_39 = shape[38]
        le_41 = shape[40]
        le_42 = shape[41]

        re_44 = shape[43]
        re_45 = shape[44]
        re_47 = shape[46]
        re_48 = shape[47]

        dlist.append((dist(le_38, le_42) + dist(le_39, le_41) + dist(re_44, re_48) + dist(re_45, re_47)) / 4 < threshold)
        if len(dlist) > 10: dlist.pop(0)

        # Drowsiness detected
        if sum(dlist) >= 4:
            try:
                sound.play()
            except:
                pass
        else:
            try:
                sound.stop()
            except:
                pass

    # Show the image
    cv2.imshow("Output", image)

    if cv2.waitKey(5) & 0xFF == 27:
        break

cv2.destroyAllWindows()
cap.release()