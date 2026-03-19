from Bibliotheken.NeuralNetworkPipeline import NeuralNetworkPipeline
import random
import os
import numpy as np
import tensorflow as tf
import pandas as pd
import keras
from keras import layers
from keras.callbacks import EarlyStopping
#-----------------------------------------------------------------------------------------------------------------------
# Für Reproduzierbarkeit
seed = int(os.getenv("PYTHONHASHSEED", 0))
# 1. Python-Seed
random.seed(seed)
# 2. NumPy-Seed
np.random.seed(seed)
# 3. TensorFlow-Seed
tf.random.set_seed(seed)
# deterministisches Verhalten auf CPU/GPU
os.environ["TF_DETERMINISTIC_OPS"] = "1"
# Informationen von TensorFlow in der Konsole ausschalten
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
#-----------------------------------------------------------------------------------------------------------------------
# Model-Definition
# Model ohne Embeddings
Eingang=keras.Input(shape=(31,))
x=layers.Dense(units=64)(Eingang)
x=layers.Activation("relu")(x)
Ausgang=layers.Dense(units=1, activation="sigmoid", name="Output-layer")(x)
KNN=keras.Model(inputs=Eingang, outputs=Ausgang)
#-----------------------------------------------------------------------------------------------------------------------
# Pipeline-Erstellung
Pipeline = NeuralNetworkPipeline('Pipeline')
Pipeline._set_model(KNN)
Pipeline._set_trainalgo(optimizer='adam',
                        loss='binary_crossentropy',
                        metrics=[])
Pipeline._set_preprocessing_options(maxcardhiddencat=15,
                                    maxcardcat=15,
                                    maxmisspercent=50,
                                    detection_excep=[],
                                    drop_excep=[],
                                    impute_excep=[],
                                    cols_for_manual_drop=["PassengerId"],
                                    cols_for_onehot=["Pclass", "Sex", "SibSp", "Parch", "Embarked"],
                                    cols_for_embedding=[],
                                    cols_and_scalers={'Fare':'Standard'}
                                    )
#-----------------------------------------------------------------------------------------------------------------------
# Training
TrainValData = pd.read_csv(os.path.join("Datensaetze", "train.csv"))
Pipeline._train(Data=TrainValData,
                       Target='Survived',
                       train_test_split_ratio=0.3,
                       batchsize=80,
                       epochs=500,
                       stopper=EarlyStopping(min_delta=0.0001,
                                             patience=80,
                                             restore_best_weights=True)
                )
Pipeline._show_trainchart()
#-----------------------------------------------------------------------------------------------------------------------
# Pipeline speichern
Pipeline._save_pipeline_in_joblib()