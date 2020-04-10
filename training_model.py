#importing libraries
import os
import time
import cv2
import numpy as np
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout

import matplotlib.image as mpimg 
import matplotlib.pyplot as plt 
from keras.utils import np_utils

#defining model
def create_model():
    model = Sequential()
    model.add(Conv2D(filters=16, kernel_size=(3,3), activation="relu", input_shape=(64,64,1)))
    model.add(Conv2D(filters=16, kernel_size=(3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu'))
    model.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(filters=64, kernel_size=(3,3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(6, activation='sigmoid'))
    model.summary()
    return model

model=create_model()

# paths to required directory
path_train='.\\Images\\image_train\\'
path_test='.\\Images\\image_test\\'

path_palm = path_train+'palm\\'
path_ok = path_train+'ok\\'
path_peace = path_train+'peace\\'
path_punch = path_train+'punch\\'
path_thumbs_up = path_train+'thumbs_up\\'
path_right = path_train+'right\\'

path_train_converted='.\\Images\\image_train_converted\\'
path_test_converted='.\\Images\\image_test_converted\\'

path_converted_palm = path_train_converted+'palm\\'
path_converted_ok = path_train_converted+'ok\\'
path_converted_peace = path_train_converted+'peace\\'
path_converted_punch = path_train_converted+'punch\\'
path_converted_thumbs_up = path_train_converted+'thumbs_up\\'
path_converted_right = path_train_converted+'right\\'


#converting color image to mask image

def color_to_mask(path_og,path_converted,im_type):
    i=0
    for files in os.listdir(path_og):
        
        
        img2=image.load_img(os.path.join(path_og+"{}_{}.png".format(im_type,i)),target_size=(64,64))
        img2=np.array(img2)

        img2=cv2.cvtColor(img2,cv2.COLOR_BGR2RGB)

        
        lower = np.array([0, 10, 60], dtype = "uint8") #for skin color
        upper = np.array([20, 150, 255], dtype = "uint8")
        hsv=cv2.cvtColor(img2,cv2.COLOR_BGR2HSV)    
        mask=cv2.inRange(hsv,lower,upper)

        
        cv2.imwrite(path_converted+'{}_{}.png'.format(im_type,i),mask)
        
        i=i+1
        cv2.waitKey(0)
        cv2.destroyAllWindows()


color_to_mask(path_palm,path_converted_palm,'palm')    
color_to_mask(path_ok,path_converted_ok,'ok')
color_to_mask(path_peace,path_converted_peace,'peace')
color_to_mask(path_thumbs_up,path_converted_thumbs_up,'thumb')
color_to_mask(path_punch,path_converted_punch,'punch')
color_to_mask(path_right,path_converted_right,'right')


result_dic={
0 : "palm",
1 : "ok",
2 : "peace",
3 : "punch",
4 : "thumbs_up",
5 : "right"
}

initial_dic={
"palm" : 0,
"ok" : 1,
"peace" : 2,
"punch" : 3,
"thumbs_up" : 4,
"right" : 5
}



dataset=[]
y=[]
i=[0]
def creating_dataset(path_enter):
    
    for files in os.listdir(path_enter):
        #img=image.load_img(files,target_size=(64,64))
        img2=cv2.imread(os.path.join(path_enter ,files ),0)
        #img2=cv2.cvtColor(img2,cv2.COLOR_RGB2GRAY)
        img_array=image.img_to_array(img2)
        dataset.append(img_array)
        path_split=path_enter.split('\\')
        y.append(initial_dic[path_split[-2].lower()])
        i[0]=i[0]+1


        
creating_dataset(path_converted_palm)
creating_dataset(path_converted_ok)
creating_dataset(path_converted_peace)
creating_dataset(path_converted_punch)
creating_dataset(path_converted_right)
creating_dataset(path_converted_thumbs_up)


dataset_array=np.array(dataset)
y=np.array(y)
dataset_array=dataset_array/255

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( dataset_array, y, test_size=0.1, random_state=0)


y_train=np_utils.to_categorical(y_train,num_classes=6)    #converting array to 6 class array
y_test=np_utils.to_categorical(y_test,num_classes=6)


#compiling and fitting the model
model.compile(optimizer="RMSprop",loss="categorical_crossentropy",metrics=["accuracy"])


#data augumentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    shear_range=0.1)

val_datagen = ImageDataGenerator() 
val_datagen.fit(X_test)
datagen.fit(X_train)

model.fit_generator(datagen.flow(X_train, y_train),
                    steps_per_epoch=len(X_train),
                    epochs=6,
                    validation_data=val_datagen.flow(X_test, y_test),
                    validation_steps=5)


#saving model to disk
from keras.models import model_from_json
model_json = model.to_json()
with open(".\\model_info\\hand_gesture_recognizer.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights(".\\model_info\\hand_gesture_recognizer.h5")
print("Saved model to disk")

