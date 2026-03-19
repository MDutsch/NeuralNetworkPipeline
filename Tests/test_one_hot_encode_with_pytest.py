# Testbeschreibung:
# - Spalten zur Encodierung sind per selected_cols auswählbar (vergleich Test1 und Test2)
# - Encoding-Kategorie-Wertemenge ist aus Einträgen der Spalte der Trainingsdaten zu generieren
#   (+'unknown','was_missing'), Werte in den Testdaten die sich nicht in dieser Menge befinden, werden
#   auf "unknown" encodiert (siehe Spalte A)
# - Kategorie-Wert 'unknown' bekommt immer eine Spalte bei Encodierung von Trainings- und Testdaten!
# - Kategorie-Wert 'was_missing' bekommt nie eine Spalte bei Encodierung von Trainings- und Testdaten!
#   (schon beim Imputing-Prozess generiert)

# Eigene Bibliotheken
from Bibliotheken.DataPreprocessingFunctions import *
# Allgemeine Bibliotheken
import pytest
import pandas as pd

def one_hot_encode(Train_Data,selected_cols,Val_Data):
    preproc_Train_Data, one_hot_encode_info=one_hot_encode_train(Train_Data,selected_cols)
    preproc_Val_Data=one_hot_encode_testval(Val_Data, one_hot_encode_info)
    return preproc_Train_Data, preproc_Val_Data


@pytest.mark.parametrize("TrainData, selected_cols, ValData, expection1,expection2",
[
(
# Test1
# Eingabe
# Trainingsdaten
pd.DataFrame({
    'A': pd.Series(['3', '2', 'was_missing', '4', '5'],
    dtype=pd.CategoricalDtype(categories=['1', '2', '3', '4', '5', '6', 'was_missing'])),
    'B': pd.Series(['2', '2', '4', '3','3'],
    dtype=pd.CategoricalDtype(categories=['4', '2', '3','was_missing'])),
    }),
# Ausgewählte Spalten für Encodierung
["A"],
# Testdaten
pd.DataFrame({
    'A': pd.Series(['3', '1', '6', '4', '1'],
    dtype=pd.CategoricalDtype(categories=['1', '3', '4', '6','was_missing'])),
    'B': pd.Series(['4', '2', '3', '4', 'was_missing'],
    dtype=pd.CategoricalDtype(categories=['2', '3', '4', 'was_missing'])),
    }),
# Erwartung
# Transformation der Trainingsdaten
pd.DataFrame({
    'B': pd.Series(['2', '2', '4', '3','3'],
    dtype=pd.CategoricalDtype(categories=['4', '2', '3','was_missing'])),
    'A_unknown': pd.Series(['0', '0', '0', '0','0'],dtype='int64'),
    'A_2': pd.Series(['0', '1', '0', '0','0'],dtype='int64'),
    'A_3': pd.Series(['1', '0', '0', '0','0'],dtype='int64'),
    'A_4': pd.Series(['0', '0', '0', '1','0'],dtype='int64'),
    'A_5': pd.Series(['0', '0', '0', '0','1'],dtype='int64')
    }),
# Transformation der Testdaten
pd.DataFrame({
    'B': pd.Series(['4', '2', '3', '4', 'was_missing'],
    dtype=pd.CategoricalDtype(categories=['2', '3', '4', 'was_missing'])),
    'A_unknown': pd.Series(['0', '1', '1', '0', '1'], dtype='int64'),
    'A_2': pd.Series(['0', '0', '0', '0', '0'], dtype='int64'),
    'A_3': pd.Series(['1', '0', '0', '0', '0'], dtype='int64'),
    'A_4': pd.Series(['0', '0', '0', '1', '0'], dtype='int64'),
    'A_5': pd.Series(['0', '0', '0', '0', '0'], dtype='int64')
})
),
(
# Test2
# Eingabe
# Trainingsdaten
pd.DataFrame({
    'A': pd.Series(['3', '2', 'was_missing', '4', '5'],
    dtype=pd.CategoricalDtype(categories=['1', '2', '3', '4', '5', '6', 'was_missing'])),
    'B': pd.Series(['2', '2', '4', '3','3'],
    dtype=pd.CategoricalDtype(categories=['4', '2', '3','was_missing'])),
    }),
# Ausgewählte Spalten für Encodierung
["A","B"],
# Testdaten
pd.DataFrame({
    'A': pd.Series(['3', '1', '6', '4', '1'],
    dtype=pd.CategoricalDtype(categories=['1', '3', '4', '6','was_missing'])),
    'B': pd.Series(['4', '2', '3', '4', 'was_missing'],
    dtype=pd.CategoricalDtype(categories=['2', '3', '4', 'was_missing'])),
    }),
# Erwartung
# Transformation der Trainingsdaten
pd.DataFrame({
    'A_unknown': pd.Series(['0', '0', '0', '0','0'],dtype='int64'),
    'A_2': pd.Series(['0', '1', '0', '0','0'],dtype='int64'),
    'A_3': pd.Series(['1', '0', '0', '0','0'],dtype='int64'),
    'A_4': pd.Series(['0', '0', '0', '1','0'],dtype='int64'),
    'A_5': pd.Series(['0', '0', '0', '0','1'],dtype='int64'),
    'B_unknown': pd.Series(['0', '0', '0', '0','0'],dtype='int64'),
    'B_2': pd.Series(['1', '1', '0', '0','0'],dtype='int64'),
    'B_3': pd.Series(['0', '0', '0', '1','1'],dtype='int64'),
    'B_4': pd.Series(['0', '0', '1', '0','0'],dtype='int64')
    }),
# Transformation der Testdaten
pd.DataFrame({
    'A_unknown': pd.Series(['0', '1', '1', '0', '1'], dtype='int64'),
    'A_2': pd.Series(['0', '0', '0', '0', '0'], dtype='int64'),
    'A_3': pd.Series(['1', '0', '0', '0', '0'], dtype='int64'),
    'A_4': pd.Series(['0', '0', '0', '1', '0'], dtype='int64'),
    'A_5': pd.Series(['0', '0', '0', '0', '0'], dtype='int64'),
    'B_unknown': pd.Series(['0', '0', '0', '0','0'],dtype='int64'),
    'B_2': pd.Series(['0', '1', '0', '0','0'],dtype='int64'),
    'B_3': pd.Series(['0', '0', '1', '0','0'],dtype='int64'),
    'B_4': pd.Series(['1', '0', '0', '1','0'],dtype='int64')
})
)
])

def test_one_hot_encode(TrainData, selected_cols, ValData,expection1,expection2):
    preproc_TrainData,preproc_TestData=one_hot_encode(TrainData,selected_cols,ValData)
    pd.testing.assert_frame_equal(preproc_TrainData, expection1)
    pd.testing.assert_frame_equal(preproc_TestData, expection2)