import cv2
from tensorflow import keras
from keras.preprocessing.image import load_img , img_to_array
import numpy as np
import create_model

model = keras.models.load_model('model.h5')#load in the mask detector model

img_width , img_height = 150,150#image reshape size

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')#load in the face detector

cap = cv2.VideoCapture(0)#start capturing the video

img_count_full = 0

font = cv2.FONT_HERSHEY_SIMPLEX
org = (1,1)
class_label = ''#will be set to mask or no mask
fontScale = 1
color = (255,0,0)
thickness = 2
#label variables

while True:
	img_count_full += 1
	response , color_img = cap.read()#captures the current frame

	if response == False:
		break


	scale = 50
	width = int(color_img.shape[1]*scale /100)
	height = int(color_img.shape[0]*scale/100)
	dim = (width,height)

	color_img = cv2.resize(color_img, dim ,interpolation= cv2.INTER_AREA)

	gray_img = cv2.cvtColor(color_img,cv2.COLOR_BGR2GRAY)

	faces = face_cascade.detectMultiScale(gray_img, 1.3, 3)#look for faces

	img_count = 0
	for (x,y,w,h) in faces:
		org = (x-10,y-10)
		img_count += 1
		color_face = color_img[y:y+h,x:x+w]
		cv2.imwrite('input/fac.jpg',color_face)
		img = load_img('input/fac.jpg',target_size=(img_width,img_height))
		img = img_to_array(img)
		img = np.expand_dims(img,axis=0)
		prediction = model.predict(img)
	#loop through each face and predict if the face has a mask on or not

		if prediction==1:
			class_label = "No Mask"
			color = (0,255,0)
		else:
			class_label = "Mask"
			color = (255,0,0)

		cv2.rectangle(color_img,(x,y),(x+w,y+h),(0,0,255),3)
		cv2.putText(color_img, class_label, org, font ,fontScale, color, thickness,cv2.LINE_AA)
		#draw the label around the face
	cv2.imshow('Face mask detection', color_img)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()
