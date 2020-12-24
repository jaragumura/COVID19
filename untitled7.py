# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 05:00:22 2020

@author: arumugaraj
"""
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import CSVLogger
#import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import GlobalAveragePooling2D
import numpy as np
#import numpy
from sklearn.metrics import confusion_matrix
import tensorflow as tf
#from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import roc_curve
#from sklearn.metrics import roc_auc_score
#from sklearn import svm, datasets
import seaborn as sns
from  tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.models import Model
MODEL_SUMMARY_FILE = "C:/Users/arumugaraj/Desktop/New folder (2)/New folder/Raj/model_summary.txt"
MODEL_FILE = "C:/Users/arumugaraj/Desktop/New folder (2)/New folder/Raj/model.h5"
TRAINING_LOGS_FILE = "training_logs.csv"
training_data_dir ="D:/New folder (2)/c_b_covid_fold5/train" 
validation_data_dir = "D:/New folder (2)/c_b_covid_fold5/test"

# Hyperparams
IMAGE_SIZE = 224
IMAGE_WIDTH, IMAGE_HEIGHT = IMAGE_SIZE, IMAGE_SIZE
EPOCHS =5
BATCH_SIZE =3
#TEST_SIZE = 3
learning = 0.00001
input_shape = (IMAGE_WIDTH, IMAGE_HEIGHT, 3)

#InceptionV3
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras import Sequential
from tensorflow.keras.applications.inception_v3 import InceptionV3
#load pre trained Xception model
base_model=tf.keras.applications.inception_v3.InceptionV3(weights='imagenet',include_top=False)
#base_model.summary()
# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = Dense(1024, activation='relu')(x)
x=Dropout(0.4)(x)
predictions = Dense(3, activation='softmax')(x)



model = Model(inputs=base_model.input, outputs=predictions)

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
for layer in base_model.layers[:-1]:
   layer.trainable = False

# compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer=Adam(lr=learning, beta_1=0.9, beta_2=0.999, amsgrad=False), loss='categorical_crossentropy',metrics=['accuracy'])
#Summary of Xception Model

#import warnings
#warnings.filterwarnings("ignore")

model.summary()



with open(MODEL_SUMMARY_FILE,"w") as fh:
    model.summary(print_fn=lambda line: fh.write(line + "\n"))

# Data augmentation
training_data_generator = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True)
validation_data_generator = ImageDataGenerator(rescale=1./255)
test_data_generator = ImageDataGenerator(rescale=1./255)

# Data preparation
training_generator = training_data_generator.flow_from_directory(
    training_data_dir,
    target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode="categorical")
validation_generator = validation_data_generator.flow_from_directory(
    validation_data_dir,
    target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False)

csv_logger = CSVLogger('training_log.csv')
import warnings
warnings.filterwarnings("ignore")



# Training
print("training")
H = model.fit_generator(
    training_generator,
    steps_per_epoch=len(training_generator.filenames) // BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=validation_generator,
    validation_steps=len(validation_generator.filenames) // BATCH_SIZE,
    callbacks=[csv_logger])

model.save_weights(MODEL_FILE)

N = EPOCHS
plt.style.use("seaborn-white")
plt.figure(dpi=600)
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="test_loss")
plt.title("Training Loss",fontsize=14)
plt.xlabel("Epoch",fontsize=14)
plt.ylabel("Loss",fontsize=14)
plt.legend(loc="upper right",fontsize=14)
plt.savefig("plot.png")

plt.style.use("seaborn-white")
plt.figure(dpi=600)
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="test_acc")
plt.title("Training Accuracy",fontsize=14)
plt.xlabel("Epoch",fontsize=14)
plt.ylabel("Accuracy",fontsize=14)
plt.legend(loc="lower right",fontsize=14)
plt.savefig("plot.png")

from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sn

LABELS = ["Nofinding","Bacterial","Covid-19"]

def show_confusion_matrix(validations, predictions):
    matrix = confusion_matrix(validations, predictions)
    plt.figure(figsize=(12, 12), dpi=600)
    sn.set(font_scale=1.6)#for label size
    sns.heatmap(matrix,
                cmap="coolwarm",
                linecolor='white',
                linewidths=1,
                xticklabels=LABELS,
                yticklabels=LABELS,
                annot=True,
                fmt="d")
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.show()

validation_generator = validation_data_generator.flow_from_directory(
    validation_data_dir,
    target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False)    

filenames = validation_generator.filenames
nb_samples = len(filenames)

Y_pred = model.predict_generator(validation_generator,(nb_samples//BATCH_SIZE)+1)
y_pred = np.argmax(Y_pred, axis=1)
print('Confusion Matrix')
show_confusion_matrix(validation_generator.classes, y_pred)
print(confusion_matrix(validation_generator.classes, y_pred))
print('Classification Report')
target_names =["Nofinding","covid","bacterial"]
print(classification_report(validation_generator.classes, y_pred, target_names=target_names))
sn.set(font_scale=1.4)#for label size



from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
plt.style.use("seaborn-white")
y_test = label_binarize(validation_generator.classes, classes=[0, 1, 2])
y_pred= label_binarize(y_pred, classes=[0, 1, 2])
n_classes = y_pred.shape[1]
# Plot linewidth.
lw = 2
# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
   fpr[i], tpr[i], _ = roc_curve(y_test[:,i], y_pred[:,i])
   roc_auc[i] = auc(fpr[i], tpr[i])

plt.figure(dpi=600)
plt.style.use("seaborn-white")
lw = 2
plt.plot(fpr[i], tpr[i], color='darkorange',
       lw=lw, label='ROC curve (area = %0.4f)' % roc_auc[i])
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate',fontsize=14)
plt.ylabel('True Positive Rate',fontsize=14)
plt.legend(loc="lower right",fontsize=14)
plt.show()




from itertools import cycle
plt.figure(2)
colors = cycle(['darkmagenta', 'darkorange', 'darkblue'])
for i, color in zip(range(1), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([-0.01, 1.0])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic to multi-class')
plt.legend(loc="lower right")
plt.show()


