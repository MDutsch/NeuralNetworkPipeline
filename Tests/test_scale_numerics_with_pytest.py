# Testbeschreibung:


# Eigene Bibliotheken
from Bibliotheken.DataPreprocessingFunctions import *
# Allgemeine Bibliothek
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import joblib as jl
import numpy as np
import pandas as pd
import pytest

def scale_numerics(TrainData,cols_and_scalers,ValData):
    TrainData, columnscalers_info=scale_numerics_train(TrainData,cols_and_scalers)
    ValData=scale_numerics_testval(ValData,columnscalers_info)
    return TrainData, columnscalers_info, ValData

@pytest.mark.parametrize("Train_Data, "
                         "cols_and_scalers, "
                         "Val_Data, "
                         "expection1, expection2,"
                         "expection3, expection4,"
,
[
(
# Test für MinMax-Scaler
# Eingabe
# Trainingsdaten
pd.DataFrame({
    'A': pd.Series([1.0, 2.0, 5.0, 4.0, 2.0],
    dtype="float64"),
    'B': pd.Series(['Alpha', 'Beta', 'Epsilon', 'Gamma', 'Beta'],
    dtype=pd.CategoricalDtype(categories=['Alpha', 'Beta', 'Gamma', 'Epsilon', 'was_missing'])),
    }),
# Skalierungswunsch
{'A': 'MinMax'},
# Testdaten
pd.DataFrame({
    'A': pd.Series([0.5, 2.0, 3.0, 2.5, 6.0],
    dtype="float64"),
    'B': pd.Series(['Kappa', 'Alpha', 'Beta', 'Gamma', 'Epsilon'],
    dtype=pd.CategoricalDtype(categories=['Alpha', 'Beta', 'Gamma', 'Epsilon', 'Kappa', 'was_missing'])),
    }),
# Erwartungen
# Skalierte Trainingsdaten
pd.DataFrame({
    'A': pd.Series([0, 0.25, 1, 0.75, 0.25],
    dtype="float64"),
    'B': pd.Series(['Alpha', 'Beta', 'Epsilon', 'Gamma', 'Beta'],
    dtype=pd.CategoricalDtype(categories=['Alpha', 'Beta', 'Gamma', 'Epsilon', 'was_missing'])),
    }),
# Skalierte Testdaten
pd.DataFrame({
    'A': pd.Series([-0.125, 0.25, 0.5, 0.375, 1.25],
    dtype="float64"),
    'B': pd.Series(['Kappa', 'Alpha', 'Beta', 'Gamma', 'Epsilon'],
    dtype=pd.CategoricalDtype(categories=['Alpha', 'Beta', 'Gamma', 'Epsilon', 'Kappa', 'was_missing'])),
    }),
# Max-Wert des MinMax-Scalers aus Trainingsdaten
5,
# Min-Wert des MinMax-Scalers aus Trainingsdaten
1,
),
(
# Test für Standard-Scaler
# Eingabe
# Trainingsdaten
pd.DataFrame({
    'A': pd.Series([10, 13, 7, 11, 9],
    dtype="float64"),
    'B': pd.Series(['Alpha', 'Beta', 'Epsilon', 'Gamma', 'Beta'],
    dtype=pd.CategoricalDtype(categories=['Alpha', 'Beta', 'Gamma', 'Epsilon', 'was_missing'])),
    }),
# Ausgewählte Spalten zum Skalieren
{'A':'Standard'},
# Testdaten
pd.DataFrame({
    'A': pd.Series([14, 5, 20, 13, 16],
    dtype="float64"),
    'B': pd.Series(['Kappa', 'Alpha', 'Beta', 'Gamma', 'Epsilon'],
    dtype=pd.CategoricalDtype(categories=['Alpha', 'Beta', 'Gamma', 'Epsilon', 'Kappa', 'was_missing'])),
    }),
# Erwartungen
# Skalierte Trainingsdaten
pd.DataFrame({
    'A': pd.Series([0, 1.5, -1.5, 0.5, -0.5],
    dtype="float64"),
    'B': pd.Series(['Alpha', 'Beta', 'Epsilon', 'Gamma', 'Beta'],
    dtype=pd.CategoricalDtype(categories=['Alpha', 'Beta', 'Gamma', 'Epsilon', 'was_missing'])),
    }),
# Skalierte Testdaten
pd.DataFrame({
    'A': pd.Series([2, -2.5, 5, 1.5, 3],
    dtype="float64"),
    'B': pd.Series(['Kappa', 'Alpha', 'Beta', 'Gamma', 'Epsilon'],
    dtype=pd.CategoricalDtype(categories=['Alpha', 'Beta', 'Gamma', 'Epsilon', 'Kappa', 'was_missing'])),
    }),
# Mittelwert des Standard-Scalers aus Trainingsdaten
10,
# Standardabweichung des MinMax-Scalers aus Trainingsdaten
2,
)
])
def test_scale_numerics(Train_Data, cols_and_scalers, Val_Data, expection1, expection2, expection3, expection4):
    scaled_TrainData,columnscalers_info,scaled_ValData=scale_numerics(Train_Data, cols_and_scalers, Val_Data)
    # Tests für korrektes Ordinal Encoding
    pd.testing.assert_frame_equal(scaled_TrainData, expection1)
    pd.testing.assert_frame_equal(scaled_ValData, expection2)
    for i in cols_and_scalers:
        if cols_and_scalers[i] == "MinMax":
            parameter1=columnscalers_info[i].data_max_
            parameter2=columnscalers_info[i].data_min_
            print("MinMaxScaler mit max-Wert ",parameter1)
            print("MinMaxScaler mit min-Wert ", parameter2)
        elif cols_and_scalers[i] == "Standard":
            parameter1=columnscalers_info[i].mean_
            parameter2 = columnscalers_info[i].scale_
            print("StandardScaler mit dem Mittelwert ", parameter1)
            print("StandardScaler mit der Standardabweichung ", parameter2)
    assert parameter1==expection3
    assert parameter2==expection4