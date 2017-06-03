import cv2
import Preprocess
import KemungkinanPlat

showstep= True


listPlat = []
# load the input image and convert it to grayscale
image = cv2.imread("img/plat-2.jpg")
imagecopy = image.copy()
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# load the cat detector Haar cascade, then detect cat faces
# in the input image
detector = cv2.CascadeClassifier('cscd/plat-80-25stage.xml')
rects = detector.detectMultiScale(gray, scaleFactor=1.04,minNeighbors=1, minSize=(75, 75))

for (i, (x, y, w, h)) in enumerate(rects):
	cv2.rectangle(imagecopy, (x, y), (x + w, y + h), (0, 0, 255), 2)
	crop_img = image[y:y+(h), x:x+(w)]
	Gray, Tresh = Preprocess.preprocess(crop_img)
	PosPlat= KemungkinanPlat.KemungkinanPlat(crop_img,x,y,w,h)
	PosPlat.imgGrayscale = Gray
	PosPlat.imgThresh = Tresh
	listPlat.append(PosPlat)
	if showstep == True:
		cv2.imshow("image"+str(i), crop_img)
# loop over the cat faces and draw a rectangle surrounding each

# show the detected cat faces
listPlat.sort(key = lambda PosPlat: PosPlat.intArea, reverse = True)
cv2.imshow("original", imagecopy)
print len(listPlat)
cv2.waitKey(0)

#########################################################3
