import cv2
import numpy as np
import os

import Preprocess
import KemungkinanPlat
import CharDetect

resultdst = "output/"
writeresult = True
showstep= True

SCALAR_BLACK = (0.0, 0.0, 0.0)
SCALAR_WHITE = (255.0, 255.0, 255.0)
SCALAR_YELLOW = (0.0, 255.0, 255.0)
SCALAR_GREEN = (0.0, 255.0, 0.0)
SCALAR_RED = (0.0, 0.0, 255.0)

listPlat = []
def main():
	print "start"
	##################### KNN TRAINING ################################3

	blnKNNTrainingSuccessful = CharDetect.loadKNNDataAndTrainKNN()
	if blnKNNTrainingSuccessful == False:
		print "\nerror: KNN traning was not successful\n"


	#################### load image #####################################
	image = cv2.imread("img/plat-11.jpg")
	imagecopy = image.copy()
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	################### load cascade classifier #########################
	detector = cv2.CascadeClassifier('cscd/plat-80-25stage.xml')
	rects = detector.detectMultiScale(gray, scaleFactor=1.03,minNeighbors=1, minSize=(75, 75))


	################## plate region detection using harr ################
	for (i, (x, y, w, h)) in enumerate(rects):
		cv2.rectangle(imagecopy, (x, y), (x + w, y + h), (0, 0, 255), 2)
		crop_img = image[y:y+(h), x:x+(w)]
		crop_img = 255-crop_img
		Gray, Tresh = Preprocess.preprocess(crop_img)
		PosPlat= KemungkinanPlat.KemungkinanPlat(crop_img,x,y,w,h)
		PosPlat.imgGrayscale = Gray
		PosPlat.imgThresh = Tresh
		listPlat.append(PosPlat)
	################# sort for the true plat ############################
	listPlat.sort(key = lambda PosPlat: PosPlat.intArea, reverse = True)
	if showstep == True:
		cv2.imshow("true-plat", listPlat[0].imgGrayscale)
		cv2.imshow("original", imagecopy)

		#print len(listPlat)



	TruePlates = CharDetect.detectCharsInPlates(listPlat[0])

	#print TruePlates.strChars
	#cv2.imshow("Result", TruePlates.imgThresh)

	print "end"

	cv2.waitKey(0)

	return


#########################################################3

if __name__ == "__main__":
    main()
