import os
import time
import cv2
import numpy as np
from keras.preprocessing import image

from keras.models import model_from_json

json_file = open('.\\model_info\\hand_gesture_recognizer.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights(".\\model_info\\hand_gesture_recognizer.h5")
print("Loaded model from disk")

result_dic={
0 : "palm",
1 : "ok",
2 : "peace",
3 : "punch",
4 : "thumbs_up",
5 : "right"
}

def process_test(img):
    lower = np.array([0, 10, 60], dtype = "uint8") #for skin color
    upper = np.array([20, 150, 255], dtype = "uint8")
        
    hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)    
    mask=cv2.inRange(hsv,lower,upper)
        
    res=cv2.bitwise_and(img,img,mask=mask)
    median=cv2.medianBlur(res,15)
    lower2=np.array([20,40,80],dtype="uint8")
    upper2=np.array([140,170,200],dtype="uint8")
    mask2=cv2.inRange(median,lower2,upper2)

    return mask



cap = cv2.VideoCapture(0)
upper_left=(0,100)
bottom_right=(250,350)


while True:
    ret,frame=cap.read()
    cv2.rectangle(img=frame,pt1=upper_left,pt2=bottom_right,color=(255,0,0),thickness=1)
    roi=frame[upper_left[1]:bottom_right[1],upper_left[0]:bottom_right[0]]
    mask = process_test(roi)
    
    
    
    img2=np.array(mask)
    resized = cv2.resize(img2, (64,64), interpolation = cv2.INTER_AREA)

    testing_image=[]

    img_array=image.img_to_array(resized)
    testing_image.append(img_array)
    testing_image=np.array(testing_image)
    testing_image = testing_image/255
    pred=loaded_model.predict(testing_image)
    res=result_dic[pred.argmax()]
    
    fonts=cv2.FONT_HERSHEY_COMPLEX_SMALL
    cv2.putText(img=frame,text=res,org=(10,370),fontFace=fonts,fontScale=1,color=(255,0,0),thickness=1)
    
    
    
    cv2.imshow('Webcam',frame)
    cv2.imshow('mask',mask)
    
    k=cv2.waitKey(1)
    if k%256 ==27:
        print('Exiting the setup....')
        break
        
cap.release()
cv2.destroyAllWindows()