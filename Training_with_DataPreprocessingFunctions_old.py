# Laden der Environment-Variablen
from dotenv import load_dotenv
import os
load_dotenv()
seed = int(os.getenv("PYTHONHASHSEED", 0))
buffered = int(os.getenv("PYTHONBUFFERED", 0))
# Eigene Bibliotheken
from Bibliotheken.DataPreprocessingFunctionsVersion_old import *
from Bibliotheken.DisplayFunctions import *
import os, random, keras, matplotlib
# Falls "Run with Python Console" in Configurations eingeschaltet ist
matplotlib.use("Qt5Agg")
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn as sk
# Informationen von TensorFlow in der Konsole ausschalten
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import tensorflow as tf
from keras import layers
from keras.callbacks import EarlyStopping
# Für Reproduzierbarkeit der Trainingsergebnisse
SEED = 1
# 1. Python-Seed
random.seed(SEED)
# 2. NumPy-Seed
np.random.seed(SEED)
# 3. TensorFlow-Seed
tf.random.set_seed(SEED)
os.environ["PYTHONHASHSEED"] = "SEED"
# Wichtig für deterministisches Verhalten auf CPU/GPU
os.environ["TF_DETERMINISTIC_OPS"] = "1"
#---------------------------------------Beginn-------------------------------------------------
Test_Data_file = "D:\\KI-Zusammenfassung\\Titanic - Machine Learning from Disaster\\test.csv"
Train_Data_file = "D:\\KI-Zusammenfassung\\Titanic - Machine Learning from Disaster\\train.csv"
TestData = pd.read_csv(Test_Data_file)
TrainValData = pd.read_csv(Train_Data_file)
# Zielvariable y festlegen und in Trainingsdatenset entfernen
y = TrainValData["Survived"]
TrainValData.drop("Survived", axis=1, inplace=True)
# Manuelle Vorververarbeitung
TrainValData.drop('PassengerId', axis=1, inplace=True)
TestData.drop('PassengerId', axis=1, inplace=True)
# Vorverarbeitung durch Pipeline
preproc_TrainValData, preproc_TestData = simple_pipeline(TrainValData, TestData,
                                                                   15, 15,
                                                                   50, ['Age'], [])
# Aufteilen des Datensatzes
preproc_TrainData, preproc_ValData, y_train, y_val = sk.model_selection.train_test_split(preproc_TrainValData, y,
                                                                                         test_size=0.3,random_state=SEED)
# Trainingsparameter
batchsize =80
epochs = 300
min_delta=0.0001
patience=40
# Modellarchitektur
units = 64
Titanic_KNN = keras.Sequential([
    layers.Dense(input_shape=(preproc_TrainData.shape[1],), units=units, name="Input-layer"),
    #layers.BatchNormalization(),
    layers.Activation("relu"),
    #layers.Dropout(0.3),
    #layers.Dense(units=units, name="HiddenLayer1"),
    #layers.BatchNormalization(),
    #layers.Activation("relu"),
    #layers.Dropout(0.3),
    #layers.Dense(units=units, name="HiddenLayer2"),
    #layers.BatchNormalization(),
    #layers.Activation("relu"),
    layers.Dense(units=1, name="Output-layer",activation="sigmoid")]
)
# Training
Stopper = EarlyStopping(min_delta=min_delta,
                        patience=patience,
                        restore_best_weights=True)
Titanic_KNN.compile(optimizer="adam", loss="binary_crossentropy", metrics=["binary_accuracy"])
train_history = Titanic_KNN.fit(preproc_TrainData,
                                y_train,
                                validation_data=(preproc_ValData, y_val),
                                batch_size=batchsize,
                                epochs=epochs,
                                callbacks=Stopper)
# Visualisierung des Trainingsverlaufs
figure_training=plot_training(Titanic_KNN, Stopper,
                                               train_history, epochs,
                                               batchsize, "Training mit EarlyStopping")
plt.show()