from LineDetection.LaneDectionModule import getLaneCurve
from TrafficSignDetection.TrafficSign_Test import equalize, grayscale, preprocessing, getClassName
import Webcam
import numpy as np
import cv2
import pickle

#################################

threshold = 0.75  # PROBABLITY THRESHOLD
font = cv2.FONT_HERSHEY_SIMPLEX

# Import the traffic sign model
pickle_in = open("TrafficSignDetection/model_trained.p", "rb")  ## rb = READ BYTE
model = pickle.load(pickle_in)

#################################

def main():
    cap = cv2.VideoCapture('test-video.mp4')
    frameCounter = 0

    while True:
        # Looping the vid
        frameCounter += 1
        if cap.get(cv2.CAP_PROP_FRAME_COUNT) == frameCounter:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frameCounter = 0

        _, img = cap.read()  # GET THE IMAGE - size (1280, 720)
        img = cv2.resize(img, (640, 360))  # RESIZE

        imgOrignal = img.copy()

        ##########################################################################################
        # Processing the Lane Curve
        curveVal = getLaneCurve(img, display=1)

        if curveVal > 0.1:
            cv2.putText(img, "Turn Right", (280, 180), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 255), 2)
        elif curveVal < -0.1:
            cv2.putText(img, "Turn Left", (280, 180), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 255), 2)
        else:
            cv2.putText(img, "Straight", (280, 180), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 255), 2)
        ##########################################################################################

        # Processing the Traffic Sign
        # PROCESS IMAGE

        img = np.asarray(imgOrignal)
        img = cv2.resize(img, (32, 32))
        img = preprocessing(img)
        cv2.imshow("Processed Image", img)
        img = img.reshape(1, 32, 32, 1)
        cv2.putText(imgOrignal, "CLASS: ", (20, 35), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(imgOrignal, "PROBABILITY: ", (20, 75), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)

        # PREDICT IMAGE
        predictions = model.predict(img)
        classIndex = model.predict_classes(img)
        probabilityValue = np.amax(predictions)

        if probabilityValue > threshold:
            # print(getCalssName(classIndex))
            cv2.putText(img, str(classIndex) + " " + str(getClassName(classIndex)), (120, 35), font, 0.75, (0, 0, 255), 2,
                        cv2.LINE_AA)
            cv2.putText(img, str(round(probabilityValue * 100, 2)) + "%", (180, 75), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)

        key = cv2.waitKey(1)
        if key == 27:
            break

if __name__ == "__main__":
    while True:
        main()

