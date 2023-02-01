import cv2
import numpy as np
import pytesseract
import csv
import requests



def empty(a):
    pass


def stackImages(scale, imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range(0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape[:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]),
                                                None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y] = cv2.cvtColor(imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank] * rows
        hor_con = [imageBlank] * rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None, scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor = np.hstack(imgArray)
        ver = hor
    return ver


def getContours(img, imgDraw, cThru=[100,100], showCanny=False, minArea=1000, filter=0, draw=True):
    imgDraw = imgDraw.copy()
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur =  cv2.GaussianBlur(imgGray, (5,5), 1 )
    imgCanny = cv2.Canny(imgBlur, cThru[0], cThru[1])
    kernel = np.array((10,10))
    imgDial = cv2.dilate(imgCanny, kernel, iterations = 1)
    imgClose = cv2.morphologyEx(imgDial, cv2.MORPH_CLOSE , kernel)

    if showCanny: cv2.imshow('Canny', imgClose)
    contours, hiearchy = cv2.findContours(imgClose, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    finalCountours = []
    for i in contours:
        area = cv2.contourArea(i)
        if area > minArea:
            peri = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, 0.02 * peri , True)
            bbox = cv2.boundingRect(approx)
            if filter > 0:
                if len(approx) == filter:
                    finalCountours.append([len(approx), area, approx, bbox, i ])

                else:
                    finalCountours.append([len(approx), area, approx, bbox , i ])

    finalCountours = sorted(finalCountours, key=lambda x: x[1], reverse=True)
    if draw:
        for con in finalCountours:
            x, y , w, h = con[3]
            cv2.rectangle(imgDraw, (x, y), (x+w, y+h), (255, 0, 255), 3)

    return imgDraw, finalCountours

def getRoi(img, Contours):
    roiList = []
    for con in contours:
        x, y , w , h  = con[3]
        roiList.append(img[y:y+h, x:x+w])
    return roiList


def roiDisplay(roiList):
    for x, roi in enumerate(roiList):
        roi = cv2.resize(roi, (0,0), None, 2, 2)
        cv2.imshow(str(x), roi)



def saveText(highlightedText):
    with open('HighlightedText.csv', 'w') as f:
        for text in highlightedText:
            f.writelines(f'\n{text}')



# path = 'Resources/text.png'
path = 'Resources/newtest.jpeg'





cv2.namedWindow("TrackBars")
# cv2.resizeWindow("TrackBars", 640, 240)
cv2.resizeWindow("TrackBars", 1000, 500)
cv2.createTrackbar("Hue Min", "TrackBars", 20, 179, empty)
cv2.createTrackbar("Hue Max", "TrackBars", 87, 179, empty)
cv2.createTrackbar("Sat Min", "TrackBars", 139, 255, empty)
cv2.createTrackbar("Sat Max", "TrackBars", 255, 255, empty)
cv2.createTrackbar("Val Min", "TrackBars", 53, 255, empty)
cv2.createTrackbar("Val Max", "TrackBars", 255, 255, empty)

# cv2.createTrackbar("Hue Min", "TrackBars", 0, 179, empty)
# cv2.createTrackbar("Hue Max", "TrackBars", 179, 179, empty)
# cv2.createTrackbar("Sat Min", "TrackBars", 0, 255, empty)
# cv2.createTrackbar("Sat Max", "TrackBars", 255, 255, empty)
# cv2.createTrackbar("Val Min", "TrackBars", 0, 255, empty)
# cv2.createTrackbar("Val Max", "TrackBars", 255, 255, empty)

while True:
    img = cv2.imread(path)
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h_min = cv2.getTrackbarPos("Hue Min", "TrackBars")
    h_max = cv2.getTrackbarPos("Hue Max", "TrackBars")
    s_min = cv2.getTrackbarPos("Sat Min", "TrackBars")
    s_max = cv2.getTrackbarPos("Sat Max", "TrackBars")
    v_min = cv2.getTrackbarPos("Val Min", "TrackBars")
    v_max = cv2.getTrackbarPos("Val Max", "TrackBars")
    print(h_min, h_max, s_min, s_max, v_min, v_max)
    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])
    mask = cv2.inRange(imgHSV, lower, upper)
    imgResult = cv2.bitwise_and(img, img, mask=mask)

    # cv2.imshow("Original",img)
    # cv2.imshow("HSV",imgHSV)
    # cv2.imshow("Mask", mask)
    cv2.imshow("Result", imgResult)


    imgContours, contours = getContours(imgResult, img, showCanny=True,
                                        minArea=1000, filter=4,
                                        cThru=[100, 150], draw= True)


    cv2.imshow("Contours", imgContours)
    # cv2.imshow("Cont", contours)

    roiList = getRoi(img, contours)
    roiDisplay(roiList)

    highlightedText = []
    for x, roi in enumerate(roiList):
        highlightedText.append(pytesseract.image_to_string(roi))
    saveText(highlightedText)
    break

    # cv2.waitKey(1)

with open('HighlightedText.csv') as file_obj:
    reader_obj = csv.reader(file_obj)
    for row in reader_obj:
        if row != []:
            url = "https://api.dictionaryapi.dev/api/v2/entries/en/" + row[0]
            print(url)
            response = requests.get(url)
            # print(response.text)
            json_response = response.json()
            try:
                meaning = json_response[0]['meanings']
                definition = meaning[0]['definitions']
                print(definition)
            except:
                definition = json_response['message']
                print(definition)

            filename = "vocab_notes.csv"
            with open(filename, 'a') as vocab:
                csvwriter = csv.writer(vocab, delimiter='\t', lineterminator='\n', )
                # csvwriter = csv.writer(vocab)
                str = row[0], definition
                csvwriter.writerow(str)
                vocab.close()













