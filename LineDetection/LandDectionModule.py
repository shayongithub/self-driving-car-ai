import cv2
import numpy as np
import utlis

curveList = []
avgVal = 10

def getLaneCurve(img, display = 2, ):
    # Display: 0 -> not display anything, 1 -> result only, 2 -> complete pipeline
    imgCopy = img.copy()
    imgResult = img.copy()

    ### Step1: Finding Lane
    imgThres = utlis.thresholding(img)

    ### Step 2: Warping Images
    hT, wT, c = img.shape
    points = utlis.valTrackbars()

    imgWarp = utlis.warpImg(imgThres, points, wT, hT)
    imgWarpPoints = utlis.drawPoints(imgCopy, points)

    ### Step 3: Finding Curve
    middlePoint, imgHist = utlis.getHistogram(imgWarp, display=True, minPer=0.25, region=4)
    curveAveragePoint, imgHist = utlis.getHistogram(imgWarp, display=True, minPer=0.45)

    curveRaw = curveAveragePoint - middlePoint# Smaller < 0 -> Left. Around 0: go straight

    ### Step 4: Averaging to have smooth transition
    curveList.append(curveRaw)
    if len(curveList) > avgVal:
        curveList.pop(0)#dont want to exceed the len of avg

    curve = int(sum(curveList)/len(curveList))

    ### Step 5: Display
    if display != 0:
        imgInvWarp = utlis.warpImg(imgWarp, points, wT, hT, inv=True)
        imgInvWarp = cv2.cvtColor(imgInvWarp, cv2.COLOR_GRAY2BGR)
        imgInvWarp[0:hT // 3, 0:wT] = 0, 0, 0
        imgLaneColor = np.zeros_like(img)
        imgLaneColor[:] = 0, 255, 0
        imgLaneColor = cv2.bitwise_and(imgInvWarp, imgLaneColor)
        imgResult = cv2.addWeighted(imgResult, 1, imgLaneColor, 1, 0)
        midY = 450
        cv2.putText(imgResult, str(curve), (wT // 2 - 80, 85), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0, 255), 3)
        cv2.line(imgResult, (wT // 2, midY), (wT // 2 + (curve * 3), midY), (255, 0, 255), 5)
        cv2.line(imgResult, ((wT // 2 + (curve * 3)), midY - 25), (wT // 2 + (curve * 3), midY + 25), (0, 255, 0), 5)
        for x in range(-30, 30):
            w = wT // 20
            cv2.line(imgResult, (w * x + int(curve // 50), midY - 10),
                     (w * x + int(curve // 50), midY + 10), (0, 0, 255), 2)
        # fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);
        # cv2.putText(imgResult, 'FPS ' + str(int(fps)), (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (230, 50, 50), 3);
    if display == 2:
        imgStacked = utlis.stackImages(0.7, ([img, imgWarpPoints, imgWarp],
                                             [imgHist, imgLaneColor, imgResult]))
        cv2.imshow('ImageStack', imgStacked)
    elif display == 1:
        cv2.imshow('Resutlt', imgResult)

    #### NORMALIZATION so the curve changes from -1 to 1
    curve = curve / 100
    # In case of error where curve excess 1 or -1
    if curve > 1:
        curve == 1
    if curve <-1:
        curve == -1

    return curve

if __name__ == '__main__':

    cap = cv2.VideoCapture('land_line_test2.mp4')
    frameCounter = 0

    # Warping Lane
    initialTrackBarValues = [111, 213, 59, 240]
    utlis.initializeTrackbars(initialTrackBarValues)

    while True:

        # Looping the vid
        frameCounter += 1
        if cap.get(cv2.CAP_PROP_FRAME_COUNT) == frameCounter:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frameCounter = 0

        _, img = cap.read()  # GET THE IMAGE - size (1280, 720)
        img = cv2.resize(img, (640, 360))  # RESIZE
        curve = getLaneCurve(img, display=2)

        if curve > 0.1:
            cv2.putText(img, "Turn Right", (280, 180), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 255), 2)
        elif curve < -0.1:
            cv2.putText(img, "Turn Left", (280, 180), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 255), 2)
        else:
            print(curve)
            cv2.putText(img, "Straight", (280, 180), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 255), 2)

        cv2.imshow("Vid", img)
        key = cv2.waitKey(1)
        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
