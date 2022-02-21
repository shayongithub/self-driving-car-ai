import cv2

#############################################
frameWidth = 640  # CAMERA RESOLUTION
frameHeight = 480
brightness = 180
##############################################

cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)
cap.set(10, brightness)
def getImg(display=False, size=[480,240]):

    _, img = cap.read()
    img = cv2.resize(img, (size[0],size[1]))
    if display:
       cv2.imshow('IMG', img)

    return img

if __name__ == '__main__':
    while True:
        img = getImg(True)