# %%
import numpy as np
import os 
import cv2
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Flatten,Dense,Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import time
import tensorflow as tf
import subprocess
import sys

# %%
def create_dataset():
    original_train_images = []
    original_train_labels = []
    path = '/app/data/agyikepek_4_osztaly/Training'

    for classes in os.listdir(path):
        for image in os.listdir(os.path.join(path,classes)):
            original_train_labels.append(classes)
            img_path = os.path.join(path,classes,image)
            #print(img_path)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (128,128))

            original_train_images.append(img)

    original_test_images = []
    original_test_labels = []
    path = '/app/data/agyikepek_4_osztaly/Testing'


    for classes in os.listdir(path):
        for image in os.listdir(os.path.join(path,classes)):
            original_test_labels.append(classes)
            img_path = os.path.join(path,classes,image)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (128,128))
            original_test_images.append(img)
    print("Successfully loaded the dataset!")
    return original_train_images, original_train_labels, original_test_images, original_test_labels


# %%
def preprocess_data(original_train_images, original_train_labels, original_test_images, original_test_labels): 
    

    train_images = np.array(original_train_images)/255.0
    test_images = np.array(original_test_images)/255.0

    lb = LabelEncoder()

    train_labels = np.array(lb.fit_transform(original_train_labels))
    test_labels = np.array(lb.transform(original_test_labels))

    mapped_classes = dict(zip(lb.classes_, range(len(lb.classes_))))
    print (mapped_classes)

    train_images.shape,test_images.shape,train_labels.shape,test_labels.shape
    print("Data preprocessing completed successfully!")
    return train_images, train_labels, test_images, test_labels, mapped_classes

# %%
def Evaluate_metrics (y_test, y_pred,classes):
    accuracy = accuracy_score(y_test,y_pred)
    class_report = classification_report(y_test,y_pred)
    cm =confusion_matrix(y_test,y_pred)

    
        
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes.keys())
    disp.plot()

    
    print(f"Accuracy score:{accuracy}")
    print(f"Class report:{class_report}")

# %%
def collapse_images(train_images, test_images):
    Collapsed_train_images = train_images.reshape(train_images.shape[0],-1)
    Collapsed_test_images = test_images.reshape(test_images.shape[0],-1)

    train_images.shape,test_images.shape
    print("Images collapesd for machine learning!")
    return Collapsed_train_images, Collapsed_test_images

# %%
def KNN_classifier(Collapsed_train_images, train_labels, Collapsed_test_images, test_labels, mapped_classes):
    print ("Step 2: Training KNN Classifier...")
    KNN = KNeighborsClassifier(n_neighbors=4)

    KNN.fit(Collapsed_train_images,train_labels)

    pred_labels =KNN.predict(Collapsed_test_images)
    print ("KNN Classifier evaluation:")
    return Evaluate_metrics(test_labels, pred_labels,mapped_classes)

# %%
def RandomForest_classifier(Collapsed_train_images, train_labels, Collapsed_test_images, test_labels, mapped_classes):
    print ("Step 3: Training Random Forest Classifier...")
    RF = RandomForestClassifier(n_estimators=300,random_state=42, max_depth=20)

    RF.fit(Collapsed_train_images,train_labels)

    pred_labels = RF.predict(Collapsed_test_images)

    print ("Random Forest Classifier evaluation:")
    return Evaluate_metrics(test_labels,pred_labels,mapped_classes)

# %%
def XGBoost_classifier(Collapsed_train_images, train_labels, Collapsed_test_images, test_labels, mapped_classes):
    print ("Step 4: Training XGBoost Classifier...")
    xg = XGBClassifier(objective="multi:softprob",eval_metric="pre", n_estimators=200, max_depth=5, learning_rate=0.07, subsample=0.85, colsample_bytree=0.9,verbosity=2, random_state=42)
    xg.fit(Collapsed_train_images,train_labels)
    pred_labels = xg.predict(Collapsed_test_images)

    print ("XGBoost Classifier evaluation:")
    return Evaluate_metrics(test_labels,pred_labels,mapped_classes)

# %%
def filter_images(original_train_images, original_test_images, train_labels):
    filterd_train_images = []

    for images in original_train_images:
        filterd_train_images.append(cv2.medianBlur(images,ksize=3))

    filtered_test_images = []

    for images in original_test_images:
        filtered_test_images.append(cv2.medianBlur(images,ksize=3))

# for i in range(5):
#     plt.subplot(5,5,i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(filterd_train_images[i],cmap="grey")
#     plt.xlabel(train_labels[i])
# plt.show()

    filterd_train_images = np.array(filterd_train_images)/255.0
    filtered_test_images = np.array(filtered_test_images)/255.0

    X =filterd_train_images
    y = train_labels


    return train_test_split(X,y, test_size=0.2, random_state=42)

# %%
def create_cnn_model():
    model = Sequential([Conv2D(32,(3,3),activation="relu", input_shape=(128,128,1)),
                        Conv2D(32,(3,3), activation="relu"),
                        MaxPooling2D(pool_size=(2,2)),
                        Dropout(0.35),
                        Conv2D(64,(3,3), activation="relu"),
                        Conv2D(64,(3,3), activation="relu"),
                        MaxPooling2D(pool_size=(2,2)),
                        Dropout(0.35),
                        Conv2D(128,(3,3), activation="relu"),
                        Conv2D(128,(3,3), activation="relu"),
                        MaxPooling2D(pool_size=(2,2)),
                        Dropout(0.3),
                        Flatten(),
                        Dense(128, activation="relu"),
                        Dropout(0.3),
                        Dense(4, activation="softmax")
                        ])
    return model

# %%
class TimeAndGpuMonitor(tf.keras.callbacks.Callback):
    def __init__(self, max_seconds=3600):
        super().__init__()
        self.max_seconds = max_seconds

    def on_train_begin(self, logs=None):
        self.start_time = time.time()
        print(f"Training started, time limit: {self.max_seconds} seconds")

    def on_epoch_end(self, epoch, logs=None):
        elapsed_time = time.time() - self.start_time

        # GPU kapcsolat ellenőrzés
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            gpu_status = "GPU detected "
            print (gpus)
        else:
            gpu_status = "No GPU detected!"

        # GPU kihasználtság lekérése (nvidia-smi parancs)
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

        # Időkorlát ellenőrzés
        if elapsed_time > self.max_seconds:
            print(f"Time limit of {self.max_seconds} seconds reached. Stopping training.")
            self.model.stop_training = True

# %%

checkpoint = ModelCheckpoint("Best_VGG.keras", monitor="val_loss",save_best_only=True,mode="min",verbose=0)
# early_stopping = EarlyStopping(monitor="val_loss", patience=7,mode="min",verbose=1)

# %%
def train_model(filterd_train_images, filtered_train_labels, val_images, val_labels, model):
    history = model.fit (filterd_train_images,filtered_train_labels, validation_data=(val_images,val_labels), epochs=10000, batch_size=32, callbacks=[checkpoint,TimeAndGpuMonitor(max_seconds=3600)], verbose=1)
    return history

# %%
def main():
    subprocess.run("clear", shell=True)
    print ("Step 1: Creating dataset...")
    original_train_images, original_train_labels, original_test_images, original_test_labels = create_dataset()
    train_images, train_labels, test_images, test_labels, mapped_classes = preprocess_data(original_train_images, original_train_labels, original_test_images, original_test_labels)
    Collapsed_train_images, Collapsed_test_images = collapse_images(train_images, test_images)
    filterd_train_images, val_images, filtered_train_labels, val_labels = filter_images(original_train_images, original_test_images, train_labels)

    if (len(sys.argv)>1 and sys.argv[1] == "manual"):
        key_input = input("Press Enter to continue with the classifiers...")
        if key_input == "":
            KNN_classifier(Collapsed_train_images, train_labels, Collapsed_test_images, test_labels, mapped_classes)
            RandomForest_classifier(Collapsed_train_images, train_labels, Collapsed_test_images, test_labels, mapped_classes)
            XGBoost_classifier(Collapsed_train_images, train_labels, Collapsed_test_images,test_labels,mapped_classes)
        else: print("Testing classifiers skipped.")
    else:
        KNN_classifier(Collapsed_train_images, train_labels, Collapsed_test_images, test_labels, mapped_classes)
        RandomForest_classifier(Collapsed_train_images, train_labels, Collapsed_test_images, test_labels, mapped_classes)
        XGBoost_classifier(Collapsed_train_images, train_labels, Collapsed_test_images,test_labels,mapped_classes)

    if (len(sys.argv)>1 and sys.argv[1] == "manual"):
        key_input = input("Press Enter ot continue with the one hour cnn train stress test...")
        if key_input == "":
            print ("Step 5: Training CNN model...")
            model = create_cnn_model()
            model.compile (
                optimizer="adam",
                loss="sparse_categorical_crossentropy",
                metrics = ["accuracy"]
            )
            history = train_model(filterd_train_images, filtered_train_labels, val_images, val_labels, model)
            test_loss, test_accuracy = model.evaluate(test_images,test_labels)
        else: print("CNN training skipped.")

    else:
        print ("Step 5: Training CNN model...")
        model = create_cnn_model()
        model.compile (
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics = ["accuracy"]
        )
        history = train_model(filterd_train_images, filtered_train_labels, val_images, val_labels, model)
        test_loss, test_accuracy = model.evaluate(test_images,test_labels)

    # plt.plot (history.history["accuracy"],label= "Train accuracy")
    # plt.plot (history.history["val_accuracy"], label = "Validation accuracy")
    # plt.title("Model accuracy")
    # plt.xlabel("Epochs")
    # plt.ylabel("Accuracy")
    # plt.legend()
    # plt.show()

if __name__ == "__main__":
    main()

# %%
# import tensorflow as tf
# from tensorflow.keras.preprocessing.image import ImageDataGenerator

# datagen = ImageDataGenerator (
#           rotation_range =5,
#           width_shift_range = 0.1,
#           height_shift_range =0.1,
#           zoom_range =0.15
# )

# train_aug_images = []
# train_aug_labels = []
# normalized_images = np.array(original_train_images)/255.0
# print (normalized_images.shape)

# augmentor= datagen.flow (normalized_images,train_labels, batch_size=len(normalized_images))

# for (x_batch, y_batch) in augmentor:
#     train_aug_images = x_batch
#     train_aug_labels = y_batch
#     break

# train_aug_images = np.array(train_aug_images)

# train_aug_images = train_aug_images.reshape(train_aug_images.shape[0],-1)
# train_aug_images.shape



# %%
# KNN = KNeighborsClassifier(n_neighbors=3)
# KNN.fit(train_aug_images,train_aug_labels)

# pred_labels =KNN.predict(test_images)

# # Evaluate_metrics(test_labels, pred_labels,mapped_classes)

# %%
# RF = RandomForestClassifier(random_state=42,n_estimators=100)

# RF.fit(train_aug_images,train_aug_labels)

# pred_labels= RF.predict(test_images)

# Evaluate_metrics(test_labels,pred_labels,mapped_classes)


