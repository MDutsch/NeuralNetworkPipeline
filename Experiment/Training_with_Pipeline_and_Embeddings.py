from Bibliotheken.NeuralNetworkPipeline import NeuralNetworkPipeline
import random
import os
import numpy as np
import tensorflow as tf
import pandas as pd
import keras
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
# Model mit Embedding für die Features Name und Ticket
InputEmbedding1=keras.Input(shape=(1,),name='Embedding1Input')
InputEmbedding2=keras.Input(shape=(1,),name='Embedding2Input')
NormalInputs=keras.Input(shape=(31,),name='NormalInputs')
OutputEmbed1=keras.layers.Embedding(input_dim=625,output_dim=5)(InputEmbedding1)
OutputEmbed2=keras.layers.Embedding(input_dim=512,output_dim=5)(InputEmbedding2)
OutputEmbed1=keras.layers.SpatialDropout1D(0.3)(OutputEmbed1)
OutputEmbed2=keras.layers.SpatialDropout1D(0.3)(OutputEmbed2)
OutputEmbed1=keras.layers.Flatten()(OutputEmbed1)
OutputEmbed2=keras.layers.Flatten()(OutputEmbed2)
x=keras.layers.concatenate([OutputEmbed1,OutputEmbed2,NormalInputs])
x=keras.layers.Dense(64)(x)
x=keras.layers.Activation('relu')(x)
Output=keras.layers.Dense(1, activation='sigmoid')(x)
KNN=keras.Model(inputs=[InputEmbedding1,InputEmbedding2,NormalInputs],outputs=Output)
#-----------------------------------------------------------------------------------------------------------------------
# Pipeline-Erstellung
Pipeline = NeuralNetworkPipeline('Pipeline_with_Embeddings')
Pipeline._set_model(KNN)
Pipeline._set_trainalgo(optimizer='adam',
                        loss='binary_crossentropy',
                        metrics=['binary_accuracy'])
Pipeline._set_preprocessing_options(maxcardhiddencat=15,
                                    maxcardcat=15,
                                    maxmisspercent=50,
                                    detection_excep=[],
                                    drop_excep=['Name','Ticket'],
                                    impute_excep=[],
                                    cols_for_manual_drop=["PassengerId"],
                                    cols_for_onehot=["Pclass", "Sex", "SibSp", "Parch", "Embarked"],
                                    cols_for_embedding=['Name','Ticket'],
                                    columns_and_scalers={'Fare':'Standard'},
                                    )
# Für Training mit Embedding
Pipeline._set_separate_inputs(separate_inputs={'Embedding1Input':'Name','Embedding2Input':'Ticket'})
#-----------------------------------------------------------------------------------------------------------------------
# Training
TrainValData = pd.read_csv(os.path.join("../Datensaetze", "train.csv"))
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