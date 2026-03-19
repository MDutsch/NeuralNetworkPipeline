# Testbeschreibung:
# Vorgegebene Spalten sollen aus Trainingsdaten entfernt werden
# Anhand der entfernten Spalten in den Trainingsdaten werden die gleichen Spalten in den Testdaten entfernt

# Eigene Bibliotheken
from Bibliotheken.DataPreprocessingFunctions import *
# Allgemeine Bibliotheken
import pytest
import pandas as pd
import numpy as np

def manual_columns_drop(Train_Data,selected_cols,Val_Data):
    Train_Data, manual_columns_drop=manual_columns_drop_train(Train_Data,selected_cols)
    Val_Data=manual_columns_drop_testval(Val_Data, manual_columns_drop)
    return Train_Data, Val_Data


@pytest.mark.parametrize("TrainData, selected_cols, ValData, expection1,expection2",
[
(
# Test1
# Eingabe
# Trainingsdaten
pd.DataFrame({
    'A': pd.Series([3, 2, np.nan, 4, 5]),
    'B': pd.Series(['2', '2', '4', '3','3'],
    dtype=pd.CategoricalDtype(categories=['4', '2', '3','was_missing'])),
    }),
# Ausgewählte Spalten für Encodierung
["A"],
# Testdaten
pd.DataFrame({
    'A': pd.Series([3, 1, 6, np.nan , 1]),
    'B': pd.Series(['4', '2', '3', '4', 'was_missing'],
    dtype=pd.CategoricalDtype(categories=['2', '3', '4', 'was_missing'])),
    }),
# Erwartung
# Transformation der Trainingsdaten
pd.DataFrame({
    'B': pd.Series(['2', '2', '4', '3','3'],
    dtype=pd.CategoricalDtype(categories=['4', '2', '3','was_missing']))
    }),
# Transformation der Testdaten
pd.DataFrame({
    'B': pd.Series(['4', '2', '3', '4', 'was_missing'],
    dtype=pd.CategoricalDtype(categories=['2', '3', '4', 'was_missing']))
})
),
])

def test_one_hot_encode(TrainData, selected_cols, ValData, expection1, expection2):
    preproc_TrainData,preproc_TestData=manual_columns_drop(TrainData,selected_cols,ValData)
    pd.testing.assert_frame_equal(preproc_TrainData, expection1)
    pd.testing.assert_frame_equal(preproc_TestData, expection2)