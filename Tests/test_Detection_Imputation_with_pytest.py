# Testbeschreibung:
# - Alle Datentypen der Spalten in Trainingsdaten richtig erkennen
# - Imputationswerte für alle Spalten in Trainingsdaten berechnen
# - Indikatorspalten nur für Spalten mit fehlenden Werten in den Trainingsdaten erstellen
# - Imputation für alle Spalten

# Eigene Bibliotheken
from Bibliotheken.DataPreprocessingFunctions import *
# Allgemeine Bibliotheken
import pytest
import numpy as np
import pandas as pd

def Detection_Imputation(Train_Data,Val_Data,maxcardhiddencat,maxcardcat,maxmisspercent,excep):
    column_info = column_detection_universal(Train_Data, maxcardhiddencat, excep)
    preproc_Train_Data, columns_to_drop = drop_columns_train(Train_Data, column_info, maxcardcat, maxmisspercent, excep)
    preproc_Train_Data, impute_and_convert_info = impute_columns_train(preproc_Train_Data, column_info, excep)
    preproc_Val_Data = drop_columns_testval(Val_Data, columns_to_drop)
    preproc_Val_Data = impute_columns_testval(preproc_Val_Data, impute_and_convert_info)
    return preproc_Train_Data, preproc_Val_Data


@pytest.mark.parametrize("maxcardhiddencat, maxcardcat, maxmisspercent, excep, expection1,expection2",
[
    (3,
    6,
    100,
    [],
    pd.DataFrame({
        'A': pd.Series([1, 0, 1, 0, 0],dtype="int64"),
        'B': pd.Series([1, 0, 0, 0, 0],dtype="int64"),
        'C': pd.Series([5, 1, 2, 4, 3],dtype="int64"),
        'D': pd.Series([5, 1, 3, 2, 4],dtype="int64"),
        'E': pd.Series(["green", "blue", "black", "black", "white"],
                       dtype=pd.CategoricalDtype(categories=['black', 'blue', 'green', 'white', 'was_missing'])),
        'F': pd.Series(["green", "blue", "was_missing", "black", "white"],
                       dtype=pd.CategoricalDtype(categories=['black', 'blue', 'green', 'white', 'was_missing'])),
        'B_was_missing': pd.Series([0, 0, 1, 0, 0],dtype="int64"),
        'D_was_missing': pd.Series([0, 0, 1, 0, 0],dtype="int64"),
        'F_was_missing': pd.Series([0, 0, 1, 0, 0],dtype="int64")
        }),
    pd.DataFrame({
        'A': pd.Series([1, 0],dtype="int64"),
        'B': pd.Series([1, 0],dtype="int64"),
        'C': pd.Series([4, 3],dtype="int64"),
        'D': pd.Series([4, 2],dtype="int64"),
        'E': pd.Series(["green", "was_missing"],
                       dtype=pd.CategoricalDtype(categories=['green','was_missing'])),
        'F': pd.Series(["green", "blue"],
                       dtype=pd.CategoricalDtype(categories=['blue', 'green', 'was_missing'])),
        'B_was_missing': pd.Series([0, 0],dtype="int64"),
        'D_was_missing': pd.Series([0, 0],dtype="int64"),
        'F_was_missing': pd.Series([0, 0],dtype="int64")
    }))
]
)

def test_Detection_Imputation(maxcardhiddencat,maxcardcat,maxmisspercent,excep,expection1,expection2):
    TrainData = pd.DataFrame({
        'A': ["true", "false", "true", "false", "false"],
        'B': ["true", "false", np.nan, "false", "false"],
        'C': [5, 1, 2, 4, 3],
        'D': [5, 1, np.nan, 2, 4],
        'E': ["green", "blue", "black", "black", "white"],
        'F': ["green", "blue", np.nan, "black", "white"]
    })
    ValData = pd.DataFrame({
        'A': ["true", np.nan],
        'B': ["true", "false"],
        'C': [4, np.nan],
        'D': [4, 2],
        'E': ["green", np.nan],
        'F': ["green", "blue"]
    })
    preproc_TrainData,preproc_TestData=Detection_Imputation(TrainData,ValData,maxcardhiddencat,maxcardcat,maxmisspercent,excep)
    pd.testing.assert_frame_equal(preproc_TrainData, expection1)
    pd.testing.assert_frame_equal(preproc_TestData, expection2)