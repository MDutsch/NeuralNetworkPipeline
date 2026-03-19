from Bibliotheken.NeuralNetworkPipeline import NeuralNetworkPipeline
import random
import os
import numpy as np
import tensorflow as tf
import joblib as jl
import pandas as pd
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
# Abgespeicherte Pipeline laden / Testdaten laden
Pipeline=jl.load('Pipeline.joblib')
TestData = pd.read_csv(os.path.join("Datensaetze", "test.csv"))
#-----------------------------------------------------------------------------------------------------------------------
# Vorhersage für die Testdaten mittels Pipeline
prediction=Pipeline._predict(TestData)
print(prediction)