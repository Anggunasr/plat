import cv2

image = cv2.imread("img/plat-11s.jpg")
imagecopy = image.copy()
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# load the cat detector Haar cascade, then detect, xml data in googledrive

detector = cv2.CascadeClassifier('cscd/plat-20-20stage.xml')
rects = detector.detectMultiScale(gray, scaleFactor=1.1,minNeighbors=2, minSize=(75, 75))

# loop over the plat and draw a rectangle surrounding each
for (i, (x, y, w, h)) in enumerate(rects):
	cv2.rectangle(imagecopy, (x, y), (x + w, y + h), (0, 0, 255), 2)
	crop_img = image[y:y+h, x:x+h]
	string = str(i)
	cv2.imshow("image"+string, crop_img)
# show the detected cat faces
cv2.imshow("Orignal", imagecopy)
cv2.waitKey(0)
