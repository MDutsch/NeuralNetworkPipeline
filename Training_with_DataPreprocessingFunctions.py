# Eigene Bibliotheken
from Bibliotheken.DataPreprocessingFunctions import *
from Bibliotheken.DataAnalysisFunctions import *
from Bibliotheken.DisplayFunctions import *
# Allgemeine Bibliotheken
import random
import keras
import os
import numpy as np
import tensorflow as tf
from keras import layers
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
# Für Reproduzierbarkeit
seed = int(os.getenv("PYTHONHASHSEED", 0))
# 1. Python-Seed
random.seed(seed)
# 2. NumPy-Seed
np.random.seed(seed)
# 3. TensorFlow-Seed
tf.random.set_seed(seed)
# Wichtig für deterministisches Verhalten auf CPU/GPU
os.environ["TF_DETERMINISTIC_OPS"] = "1"
# Informationen von TensorFlow in der Konsole ausschalten
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
#------------------------------------------------Beginn-----------------------------------------------------------
Test_Data_file = "D:\\KI-Zusammenfassung\\Titanic - Machine Learning from Disaster\\test.csv"
Train_Data_file = "D:\\KI-Zusammenfassung\\Titanic - Machine Learning from Disaster\\train.csv"
TestData = pd.read_csv(Test_Data_file)
TrainValData = pd.read_csv(Train_Data_file)
y = TrainValData["Survived"]
TrainValData.drop("Survived", axis=1, inplace=True)
TrainValData.drop("PassengerId", axis=1, inplace=True)
TestData.drop("PassengerId", axis=1, inplace=True)
# Datenvorverarbeitung mit Berechnung der Mutual-Information dazwischen
column_info = column_detection(TrainValData, TestData, 15, [])
preproc_TrainValData, preproc_TestData = impute_columns(TrainValData, TestData, column_info,[])
# Berechnung der Mutual-Information
figure_mutualinfo = roughly_mutual_information(preproc_TrainValData, y, classification=True, seed=seed)
# Embedding Encoding für kategorische Features mit zu hoher Kardinalzahl
for i in ["Name", "Ticket"]:
    preproc_TrainValData, preproc_TestData = encode_with_embedding(preproc_TrainValData,
                                                                   preproc_TestData, i)
# One Hot Encoding für kategorische Features mit Kardinalzahl < 15
for j in ["Pclass", "Sex", "SibSp", "Parch", "Embarked"]:
    preproc_TrainValData, preproc_TestData = one_hot_encode(preproc_TrainValData, preproc_TestData, j)
# entferne Features
preproc_TrainValData.drop(['Cabin', 'Cabin_was_missing'], axis=1, inplace=True)
preproc_TestData.drop(['Cabin', 'Cabin_was_missing'], axis=1, inplace=True)
# skaliere Features
preproc_TrainValData, preproc_TestData = scale_columns(preproc_TrainValData,preproc_TestData,
                                                       "Standard", "Age")
# Aufteilen des Datensatzes
preproc_TrainData, preproc_ValData, y_train, y_val = train_test_split(preproc_TrainValData, y,
                                                                        test_size=0.3,random_state=seed)
# Trainingsparameter
batchsize =80
epochs = 300
min_delta=0.0001
patience=40
# Modellarchitektur
units = 64
Titanic_KNN = keras.Sequential([
    layers.Input(shape=(preproc_TrainData.shape[1],)),
    layers.Dense(units=units, name="Input-layer"),
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
    #layers.Dropout(0.3),
    layers.Dense(units=1, activation="sigmoid", name="Output-layer")]
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
figure_training=plot_training(Titanic_KNN,Stopper, train_history,epochs,
                              batchsize,"Training mit EarlyStopping")
plt.show()