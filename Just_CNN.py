import numpy as np
import os 
import cv2
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Flatten,Dense,Dropout
from tensorflow.keras.layers import BatchNormalization
import time
import tensorflow as tf
import subprocess
import sys

#Function to crate dataset from training and testing folders
#The images color channels reduced to grayscale for simplicity and resized to 256x256 pixels
def create_dataset():
#Training data                  
    original_train_images = []
    original_train_labels = []
    path = '/app/data/agyikepek_4_osztaly/Training'

    for classes in os.listdir(path):
        for image in os.listdir(os.path.join(path,classes)):
            original_train_labels.append(classes)
            img_path = os.path.join(path,classes,image)
            #print(img_path)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (256,256))

            original_train_images.append(img)
    #Testing data 
    original_test_images = []
    original_test_labels = []
    path = '/app/data/agyikepek_4_osztaly/Testing'


    for classes in os.listdir(path):
        for image in os.listdir(os.path.join(path,classes)):
            original_test_labels.append(classes)
            img_path = os.path.join(path,classes,image)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (256,256))
            original_test_images.append(img)
    print("Successfully loaded the dataset!")
    return original_train_images, original_train_labels, original_test_images, original_test_labels

def preprocess_data(original_train_labels, original_test_labels): 
    
    lb = LabelEncoder()
#Encoding the labels (converting string labels to integers)

    train_labels = np.array(lb.fit_transform(original_train_labels))
    test_labels = np.array(lb.transform(original_test_labels))

    mapped_classes = dict(zip(lb.classes_, range(len(lb.classes_))))
    print (mapped_classes)

    print("Data preprocessing completed successfully!")
    return train_labels, test_labels, mapped_classes

#Function to apply median filter to the images with a kernel size of 3x3 and normalize the pixel values to the range [0, 1]
def filter_images(original_train_images, original_test_images, train_labels):
    filterd_train_images = []
#Applying median filter to the training images
    for images in original_train_images:
        filterd_train_images.append(cv2.medianBlur(images,ksize=3))

    filtered_test_images = []
#Applying median filter to the testing images
    for images in original_test_images:
        filtered_test_images.append(cv2.medianBlur(images,ksize=3))


#Normalizing the pixel values to the range [0, 1]
    filterd_train_images = np.array(filterd_train_images,dtype=np.float32)/255.0
    filtered_test_images = np.array(filtered_test_images,dtype=np.float32)/255.0

    X =filterd_train_images
    y = train_labels

    filterd_train_images, val_images, filtered_train_labels, val_labels = train_test_split(X,y, test_size=0.2, random_state=42)


    return filterd_train_images, val_images, filtered_train_labels, val_labels , filtered_test_images

def create_cnn_model():
    model = Sequential([Conv2D(32,(3,3),activation="relu", input_shape=(256,256,1)),
                        BatchNormalization(),
                        Conv2D(32,(3,3), activation="relu"),
                        MaxPooling2D(pool_size=(2,2)),
                        Dropout(0.25),
                        Conv2D(64,(3,3), activation="relu"),
                        BatchNormalization(),
                        Conv2D(64,(3,3), activation="relu"),
                        MaxPooling2D(pool_size=(2,2)),
                        Dropout(0.25),
                        Conv2D(128,(3,3), activation="relu"),
                        BatchNormalization(),
                        Conv2D(128,(3,3), activation="relu"),
                        MaxPooling2D(pool_size=(2,2)),
                        Flatten(),
                        Dense(128, activation="relu"),
                        Dropout(0.2),
                        Dense(4, activation="softmax")
                        ])
    return model

class TimeAndGpuMonitor(tf.keras.callbacks.Callback):
#Initialize with maximum training time in seconds
    def __init__(self, max_seconds=3600):
        super().__init__()
        self.max_seconds = max_seconds

    def on_train_begin(self, logs=None):
        self.start_time = time.time()
        print(f"Training started, time limit: {self.max_seconds} seconds")

    def on_epoch_end(self, epoch, logs=None):
        elapsed_time = time.time() - self.start_time

 # GPU connection check
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            gpu_status = "GPU detected "
            print (gpus)
        else:
            gpu_status = "No GPU detected!"

# GPU usage check using nvidia-smi
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                 "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True
            )
            usage = result.stdout.strip()
            if usage:
                util, mem = usage.split(",")
                gpu_info = f"GPU Usage: {util.strip()}%, Memory Used: {mem.strip()} MB"
            else:
                gpu_info = "No usage info available"
        except FileNotFoundError:
            gpu_info = "nvidia-smi not found (no NVIDIA GPU driver?)"

        print(f"--- Epoch {epoch+1} End ---")
        print(f"Elapsed Time: {elapsed_time:.2f} sec")
        print(f"{gpu_status} | {gpu_info}")

# Check if time limit exceeded
        if elapsed_time > self.max_seconds:
            print(f"Time limit of {self.max_seconds} seconds reached. Stopping training.")
            self.model.stop_training = True

def train_model(filterd_train_images, filtered_train_labels, val_images, val_labels, model):
    history = model.fit (filterd_train_images,filtered_train_labels, validation_data=(val_images,val_labels), epochs=10000, batch_size=64, callbacks=[TimeAndGpuMonitor(max_seconds=3600)], verbose=2)
    return history

def main():
    subprocess.run("clear", shell=True)
    print ("Step 1: Creating dataset...")
    original_train_images, original_train_labels, original_test_images, original_test_labels = create_dataset()
    train_labels,  test_labels, mapped_classes = preprocess_data( original_train_labels, original_test_labels)
    filterd_train_images, val_images, filtered_train_labels, val_labels, test_images = filter_images(original_train_images, original_test_images, train_labels)
    print ("Step 2: Training CNN model...")
    model = create_cnn_model()
    model.compile (
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics = ["accuracy"]
    )
    history = train_model(filterd_train_images, filtered_train_labels, val_images, val_labels, model)
    test_loss, test_accuracy = model.evaluate(test_images,test_labels, batch_size=64)
    print(f"test_loss: {test_loss}, test_accuracy: {test_accuracy}")

if __name__ == "__main__":
    main()