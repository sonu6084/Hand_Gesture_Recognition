import cv2
import os
import numpy as np 


#data collection
def display_info(arg):
    button={
    0:"     ",
    1:'palm_'+str(palm_counter),
    2:'ok_'+str(ok_counter),
    3:'peace_'+str(peace_counter),
    4:'punch_'+str(punch_counter),
    5:'thumb_'+str(thumb_counter),
    6:'right_'+str(right_counter)
    }
    return button.get(arg)


#paths for training folder
path_train='.\\Images\\image_train\\'
path_test='.\\Images\\image_test\\'
path_palm = path_train+'palm\\'
path_ok = path_train+'ok\\'
path_peace = path_train+'peace\\'
path_punch = path_train+'punch\\'
path_thumb = path_train+'thumbs_up\\'
path_right = path_train+'right\\'

#image numbers counters in directory
palm_counter=0
ok_counter=0
peace_counter=0
punch_counter=0
thumb_counter=0
right_counter=0

#number of images present in directory
for files in os.listdir(path_palm):
    palm_counter=palm_counter+1
    
for files in os.listdir(path_ok):
    ok_counter=ok_counter+1
    
for files in os.listdir(path_peace):
    peace_counter=peace_counter+1
    
for files in os.listdir(path_punch):
    punch_counter=punch_counter+1
    
for files in os.listdir(path_thumb):
    thumb_counter=thumb_counter+1
    
for files in os.listdir(path_right):
    right_counter=right_counter+1




cam=cv2.VideoCapture(0)
button_pressed=0



upper_left=(0,100)
bottom_right=(250,350)
while True:
    ret,frame=cam.read()
    cv2.rectangle(img=frame,pt1=upper_left,pt2=bottom_right,color=(255,0,0),thickness=1)
    
    
    show_text=display_info(button_pressed)
    
    fonts=cv2.FONT_HERSHEY_COMPLEX_SMALL
    cv2.putText(img=frame,text=show_text,org=(10,370),fontFace=fonts,fontScale=1,color=(255,0,0),thickness=1)
    
    cv2.imshow('Webcam',frame)
    roi=frame[upper_left[1]:bottom_right[1],upper_left[0]:bottom_right[0]]
    cv2.imshow('roi',roi)
    
    lower = np.array([0, 10, 60], dtype = "uint8") #for skin color
    upper = np.array([20, 150, 255], dtype = "uint8")
    hsv=cv2.cvtColor(roi,cv2.COLOR_BGR2HSV)    
    mask=cv2.inRange(hsv,lower,upper)
        
    res=cv2.bitwise_and(roi,roi,mask=mask)
    median=cv2.medianBlur(res,15)
    lower2=np.array([20,40,80],dtype="uint8")
    upper2=np.array([140,170,200],dtype="uint8")
    mask2=cv2.inRange(median,lower2,upper2)
    res2=cv2.bitwise_and(median,median,mask=mask2)
        
        
    cv2.imshow('mask',mask)
    cv2.imshow('mask2',mask2)
    
   
    if not ret:
        break
    
    k=cv2.waitKey(1)
    if k%256 ==27:
        print('Exiting the setup....')
        break
    elif k%256 ==ord('1'):
        cv2.imwrite(path_palm+'palm_{}.png'.format( palm_counter),roi)
        
        palm_counter=palm_counter+1
        print("written in disk-palm")
        button_pressed=1
    elif k%256 ==ord('2'):
        cv2.imwrite(path_ok+'ok_{}.png'.format( ok_counter),roi)
        ok_counter=ok_counter+1
        print("written in disk-ok")
        button_pressed=2
    elif k%256 ==ord('3'):
        cv2.imwrite(path_peace+'peace_{}.png'.format( peace_counter),roi)
        peace_counter=peace_counter+1
        print("written in disk-peace")
        button_pressed=3
    elif k%256 ==ord('4'):
        cv2.imwrite(path_punch+'punch_{}.png'.format( punch_counter),roi)
        punch_counter=punch_counter+1
        print("written in disk-punch")
        button_pressed=4
    elif k%256 ==ord('5'):
        cv2.imwrite(path_thumb+'thumb_{}.png'.format( thumb_counter),roi)
        thumb_counter=thumb_counter+1
        print("written in disk-thumb")
        button_pressed=5
    elif k%256 ==ord('6'):
        cv2.imwrite(path_right+'right_{}.png'.format( right_counter),roi)
        right_counter=right_counter+1
        print("written in disk-right")
        button_pressed=6

cam.release()
cv2.destroyAllWindows()