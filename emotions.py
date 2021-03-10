import numpy as np
import argparse
import cv2
import pickle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import matplotlib.pyplot as plt
from contextlib import contextmanager
import dlib
from keras.utils.data_utils import get_file
from wide_resnet import WideResNet
from pathlib import Path
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# command line argument
ap = argparse.ArgumentParser()
ap.add_argument("--mode",help="train/display/Age_Gender")
ap.add_argument("--image_dir",help="target image directory; if set, images in image_dir are used instead of webcam")
a = ap.parse_args()
mode = a.mode 
image_dir = a.image_dir
#parser.add_argument("--image_dir", type=str, default=None,help="target image directory; if set, images in image_dir are used instead of webcam")
'''p = argparse.ArgumentParser()
ap.add_argument("--image_dir",type=str, help="target image directory; if set, images in image_dir are used instead of webcam")
args = ap.parse_args()'''
pretrained_model = "D:\BIO\Emotion-detection-master\Emotion-detection-master\Tensorflow\\weights.28-3.73.hdf5"

modhash = 'fbe63257a054c1c5466cfd7bf14646d6'
def plot_model_history(model_history):
    """
    Plot Accuracy and Loss curves given the model_history
    """
    fig, axs = plt.subplots(1,2,figsize=(15,5))
    # summarize history for accuracy
    axs[0].plot(range(1,len(model_history.history['acc'])+1),model_history.history['acc'])
    axs[0].plot(range(1,len(model_history.history['val_acc'])+1),model_history.history['val_acc'])
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].set_xticks(np.arange(1,len(model_history.history['acc'])+1),len(model_history.history['acc'])/10)
    axs[0].legend(['train', 'val'], loc='best')
    # summarize history for loss
    axs[1].plot(range(1,len(model_history.history['loss'])+1),model_history.history['loss'])
    axs[1].plot(range(1,len(model_history.history['val_loss'])+1),model_history.history['val_loss'])
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_xticks(np.arange(1,len(model_history.history['loss'])+1),len(model_history.history['loss'])/10)
    axs[1].legend(['train', 'val'], loc='best')
    fig.savefig('plot.png')
    plt.show()

# Define data generators
train_dir = 'data/train'
#train_dir = 'data1'
val_dir = 'data/test'
#val_dir = 'data1'
train_dir_image = 'images'

num_train = 28709
num_val = 7178
#num_train=50
#num_val=12
batch_size = 64
#batch_size=8
#num_epoch=4
num_epoch = 50

train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(48,48),
        batch_size=batch_size,
        color_mode="grayscale",
        class_mode='categorical')

train_generator_image = train_datagen.flow_from_directory(
        train_dir_image,
        target_size=(48,48),
        batch_size=8,
        class_mode='categorical')


validation_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=(48,48),
        batch_size=batch_size,
        color_mode="grayscale",
        class_mode='categorical')

validation_generator_image = val_datagen.flow_from_directory(
        train_dir_image,
        target_size=(48,48),
        batch_size=8,
        class_mode='categorical')

# Create the model
model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))

#create model2

model2 = Sequential()

model2.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,3)))
model2.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model2.add(MaxPooling2D(pool_size=(2, 2)))
model2.add(Dropout(0.25))

model2.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model2.add(MaxPooling2D(pool_size=(2, 2)))
model2.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model2.add(MaxPooling2D(pool_size=(2, 2)))
model2.add(Dropout(0.25))

model2.add(Flatten())
model2.add(Dense(1024, activation='relu'))
model2.add(Dropout(0.5))
model2.add(Dense(7, activation='softmax'))

## Capturing video image
@contextmanager
def video_capture(*args, **kwargs):
    cap = cv2.VideoCapture(*args, **kwargs)
    try:
        yield cap
    finally:
        cap.release()

def yield_images():
    # capture video
    with video_capture(0) as cap:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        while True:
            # get video frame
            ret, img = cap.read()

            if not ret:
                raise RuntimeError("Failed to capture image")

            yield img

## Drwaing rectangle and labeling
def draw_label(image, point, label, font=cv2.FONT_HERSHEY_SIMPLEX,
               font_scale=0.8, thickness=1):
    size = cv2.getTextSize(label, font, font_scale, thickness)[0]
    x, y = point
    cv2.rectangle(image, (x, y - size[1]), (x + size[0], y), (255, 0, 0), cv2.FILLED)
    cv2.putText(image, label, point, font, font_scale, (255, 255, 255), thickness, lineType=cv2.LINE_AA)

## Image from directory
def yield_images_from_dir(image_dir):
    image_dir = Path(image_dir)

    for image_path in image_dir.glob("*.*"):
        img = cv2.imread(str(image_path), 1)

        if img is not None:
            h, w, _ = img.shape
            r = 640 / max(w, h)
            yield cv2.resize(img, (int(w * r), int(h * r)))

# If you want to train emotion detection module

if mode == "train":
    model.compile(loss='categorical_crossentropy',optimizer=Adam(lr=0.0001, decay=1e-6),metrics=['accuracy'])

    model_info = model.fit_generator(
            train_generator,
            steps_per_epoch=num_train // batch_size,
            epochs=num_epoch,
            validation_data=validation_generator,
            validation_steps=num_val // batch_size)

    plot_model_history(model_info)
    model.save_weights('model.h5')

elif mode == "train_image":
    model2.compile(loss='categorical_crossentropy',optimizer=Adam(lr=0.0001, decay=1e-6),metrics=['accuracy'])
    batch_size=8
    num_train=106
    model_info = model2.fit_generator(
            train_generator_image,
            steps_per_epoch=13.25,
            epochs=4,
            validation_data=validation_generator_image,
            validation_steps=13.25)

    plot_model_history(model_info)
    model2.save_weights('image.h5')

# emotions will be displayed on your face from the webcam feed or image directory
elif mode == "display":
    weight_file = get_file("weights.28-3.73.hdf5", pretrained_model,cache_subdir="pretrained_models",file_hash=modhash, cache_dir=str(Path(__file__).resolve().parent))
    # for face detection
    detector = dlib.get_frontal_face_detector()
    # load model and weights
    img_size = 64
    
    model1 = WideResNet(img_size, depth=16, k=8)()
    model1.load_weights(weight_file)




    model.load_weights('model.h5')

    # prevents openCL usage and unnecessary logging messages
    cv2.ocl.setUseOpenCL(False)

    # dictionary which assigns each label an emotion (alphabetical order)
    emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

    # start the webcam feed
    recognizer = cv2.face.LBPHFaceRecognizer_create(radius=2)
    recognizer.read("face-trainner.yml")

    labels = {"person_name": 1}

    with open("face-labels.pickle", 'rb') as f:
	    og_labels = pickle.load(f)
	    labels = {v:k for k,v in og_labels.items()}

    #path="D:\BIO\Emotion-detection-master\Emotion-detection-master\Tensorflow\image"
    
    #image_generator=yield_images_from_dir(image_dir)
    image_generator = yield_images_from_dir(image_dir) if image_dir else yield_images()
    margin=0.4
    #cap = cv2.VideoCapture(0)
    #while True:
    for frame in image_generator:
        # Find haar cascade to draw bounding box around face
        #ret, frame = cap.read()
        #if not ret:
        #    break
        img=frame
        facecasc = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = facecasc.detectMultiScale(gray,scaleFactor=1.3, minNeighbors=5)

        
        input_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_h, img_w, _ = np.shape(input_img)

        # detect faces using dlib detector
        detected = detector(input_img, 1)
        faces1 = np.empty((len(detected), img_size, img_size, 3))
        for i, d in enumerate(detected):
            x1, y1, x2, y2, w1, h1 = d.left(), d.top(), d.right() + 1, d.bottom() + 1, d.width(), d.height()
            xw1 = max(int(x1 - margin * w1), 0)
            yw1 = max(int(y1 - margin * h1), 0)
            xw2 = min(int(x2 + margin * w1), img_w - 1)
            yw2 = min(int(y2 + margin * h1), img_h - 1)
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            #cv2.rectangle(img, (xw1, yw1), (xw2, yw2), (255, 0, 0), 2)
            faces1[i, :, :, :] = cv2.resize(img[yw1:yw2 + 1, xw1:xw2 + 1, :], (img_size, img_size))

            roi_gray = gray[y1:y1+h1, x1:x1+w1]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
            prediction = model.predict(cropped_img)
            maxindex = int(np.argmax(prediction))

            
            
            id_, conf = recognizer.predict(roi_gray)
            if conf>=40 and conf<=140:
                name=labels[id_]
                color=(255,255,255)
                font=cv2.FONT_HERSHEY_COMPLEX
                stroke=2

                cv2.putText(frame,name,(x1,y1), font, 1, color, stroke, cv2.LINE_AA)
            else:
                name='Unknown'
                color=(255,255,255)
                font=cv2.FONT_HERSHEY_COMPLEX
                stroke=2
                cv2.putText(frame,name,(x1,y1), font, 1, color, stroke, cv2.LINE_AA)

        # predict ages and genders of the detected faces
        results = model1.predict(faces1)
        predicted_genders = results[0]
        ages = np.arange(0, 101).reshape(101, 1)
        predicted_ages = results[1].dot(ages).flatten()
        for i, d in enumerate(detected):
            label = "{}, {}, {}".format(str(emotion_dict[maxindex]),int(predicted_ages[i]),
                                        "M" if predicted_genders[i][0] < 0.5 else "F")

        color=(255,0,0)

        cv2.putText(frame,label , (x1+40, y1-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        
        



        '''for (x, y, w, h) in faces:
            
            xw2 = max(int(x - margin * w), img_w - 1)
            yh2 = max(int(y - margin * h), img_h - 1)
            xw1 = max(int(x - margin * w), 0)
            yh1 = max(int(y - margin * h), 0)
            #roi_gray = gray[y:y + h, x:x + w]
            #cv2.rectangle(frame, (x, y-10), (x+w, y+h+10), (255, 0, 0), 2)
            cv2.rectangle(frame, (xw1, yh1), (xw2, yh2), (255, 0, 0), 2)
            roi_gray = gray[yh1:yh2, xw1:xw2]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
            prediction = model.predict(cropped_img)
            maxindex = int(np.argmax(prediction))
            
            
            

            for i, d in enumerate(detected):
                label = "{}, {}, {}".format(str(emotion_dict[maxindex]),int(predicted_ages[i]),
                                        "M" if predicted_genders[i][0] < 0.5 else "F")

            color=(255,0,0)

            cv2.putText(frame,label , (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            
            
            
            id_, conf = recognizer.predict(roi_gray)
            if conf>=60 and conf<=120:
                name=labels[id_]
                color=(255,255,255)
                font=cv2.FONT_HERSHEY_COMPLEX
                stroke=2

                cv2.putText(frame,name,(x,y), font, 1, color, stroke, cv2.LINE_AA)
            else:
                name='Unknown'
                color=(255,255,255)
                font=cv2.FONT_HERSHEY_COMPLEX
                stroke=2
                cv2.putText(frame,name,(x,y), font, 1, color, stroke, cv2.LINE_AA)'''
        
    
        #cv2.imshow('Video', cv2.resize(frame,(1600,960),interpolation = cv2.INTER_CUBIC))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        cv2.imshow("result", frame)
        key = cv2.waitKey(-1) 

        if key == 27:  # ESC
            break

    #cap.release()
    cv2.destroyAllWindows()

elif mode == "Age_Gender":
    
    weight_file = get_file("weights.28-3.73.hdf5", pretrained_model,cache_subdir="pretrained_models",file_hash=modhash, cache_dir=str(Path(__file__).resolve().parent))
    # for face detection
    detector = dlib.get_frontal_face_detector()
    # load model and weights
    img_size = 64
    
    model = WideResNet(img_size, depth=16, k=8)()
    model.load_weights(weight_file)
    

    image_generator = yield_images_from_dir(image_dir) if image_dir else yield_images()
    
    margin=0.4
    for img in image_generator:
        input_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_h, img_w, _ = np.shape(input_img)

        # detect faces using dlib detector
        detected = detector(input_img, 1)
        faces = np.empty((len(detected), img_size, img_size, 3))
       

        if len(detected) > 0:
            for i, d in enumerate(detected):
                x1, y1, x2, y2, w, h = d.left(), d.top(), d.right() + 1, d.bottom() + 1, d.width(), d.height()
                xw1 = max(int(x1 - margin * w), 0)
                yw1 = max(int(y1 - margin * h), 0)
                xw2 = min(int(x2 + margin * w), img_w - 1)
                yw2 = min(int(y2 + margin * h), img_h - 1)
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                # cv2.rectangle(img, (xw1, yw1), (xw2, yw2), (255, 0, 0), 2)
                faces[i, :, :, :] = cv2.resize(img[yw1:yw2 + 1, xw1:xw2 + 1, :], (img_size, img_size))

            # predict ages and genders of the detected faces
            results = model.predict(faces)
            predicted_genders = results[0]
            ages = np.arange(0, 101).reshape(101, 1)
            predicted_ages = results[1].dot(ages).flatten()

            # draw results
            for i, d in enumerate(detected):
                label = "{}, {}".format(int(predicted_ages[i]),
                                        "M" if predicted_genders[i][0] < 0.5 else "F")
                draw_label(img, (d.left(), d.top()), label)

        cv2.imshow("result", img)
        key = cv2.waitKey(-1) if image_dir else cv2.waitKey(30)

        if key == 27:  # ESC
            break

elif mode == "emotion":
   
    # for face detection
    detector = dlib.get_frontal_face_detector()
    # load model and weights
    img_size = 64
    
    model.load_weights('model.h5')

    # prevents openCL usage and unnecessary logging messages
    cv2.ocl.setUseOpenCL(False)

    # dictionary which assigns each label an emotion (alphabetical order)
    emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

    # start the webcam feed
    image_generator = yield_images_from_dir(image_dir) if image_dir else yield_images()
    margin=0.4
    #cap = cv2.VideoCapture(0)
    #while True:
    for frame in image_generator:
        
        img=frame
        facecasc = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = facecasc.detectMultiScale(gray,scaleFactor=1.3, minNeighbors=5)

        
        input_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_h, img_w, _ = np.shape(input_img)

        # detect faces using dlib detector
        detected = detector(input_img, 1)
        faces1 = np.empty((len(detected), img_size, img_size, 3))
        for i, d in enumerate(detected):
            x1, y1, x2, y2, w1, h1 = d.left(), d.top(), d.right() + 1, d.bottom() + 1, d.width(), d.height()
            xw1 = max(int(x1 - margin * w1), 0)
            yw1 = max(int(y1 - margin * h1), 0)
            xw2 = min(int(x2 + margin * w1), img_w - 1)
            yw2 = min(int(y2 + margin * h1), img_h - 1)
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            #cv2.rectangle(img, (xw1, yw1), (xw2, yw2), (255, 0, 0), 2)
            

            roi_gray = gray[y1:y1+h1, x1:x1+w1]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
            prediction = model.predict(cropped_img)
            maxindex = int(np.argmax(prediction))

            
           
        cv2.putText(frame,emotion_dict[maxindex] , (x1+60, y1-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        
        #cv2.imshow('Video', cv2.resize(frame,(1600,960),interpolation = cv2.INTER_CUBIC))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        cv2.imshow("result", frame)
        key = cv2.waitKey(-1) 

        if key == 27:  # ESC
            break

    #cap.release()
    cv2.destroyAllWindows()