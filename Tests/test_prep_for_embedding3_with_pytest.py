# Testbeschreibung:
# => Spezialfall: Encodierung einer kategorischen Variable mit Einträgen aus Integern und Strings
# Encodierung eines Features mit Integer-Einträgen und 'NaN' für fehlende Werte.
# Feature wird als kategorische Variable klassifiziert und dementsprechend transformiert.
# NaN -> 'was_missing' => kategorische Variable mit Integer- und String-Einträgen
# Feature wird als kategorische Variable mit dem Ordinal-Encoder für ein Embedding encodiert.
# Encodiertes Feature wird durch die der Einträge entsprechenden Embedding-Vektoren ersetzt.

# Eigene Bibliotheken
from Bibliotheken.DataPreprocessingFunctions import *
# Allgemeine Bibliothek
from sklearn.preprocessing import OrdinalEncoder
import numpy as np
import pandas as pd
import pytest

def transform_to_category(Data,columns):
    for col in columns:
        Data[col] = Data[col].astype('category')
        Data[col] = Data[col].cat.add_categories(["was_missing"])
        Data[col] = Data[col].fillna('was_missing')
    return Data

def assign_embedding_vektor(Data,columns,output_dims):
    embedding_matrix=[[0,0],
                      [0,1],
                      [1,0],
                      [1,1],
                      [0,2],
                      [2,0]]
    for col in columns:
        Vektors=[]
        for i in Data[col]:
            Vektors.append(embedding_matrix[i])
        for i in range(0, output_dims[col]):
            Data[col+'_'+str(i)] = pd.Series(np.array(Vektors)[:,i].tolist()).astype('int64')
        Data.drop([col], axis=1, inplace=True)
    return Data

def prep_for_embedding(Train_Data, selected_cols, Val_Data):
    transf_Train_Data = transform_to_category(Train_Data, selected_cols)
    encoded_Train_Data, embedding_info ,input_dim, output_dim=prep_for_embedding_train(transf_Train_Data, selected_cols)
    transf_Val_Data = transform_to_category(Val_Data, selected_cols)
    encoded_Val_Data=prep_for_embedding_testval(transf_Val_Data, embedding_info)
    preproc_Train_Data=assign_embedding_vektor(encoded_Train_Data.copy(), selected_cols,output_dim)
    preproc_Val_Data=assign_embedding_vektor(encoded_Val_Data.copy(), selected_cols, output_dim)
    return (encoded_Train_Data, encoded_Val_Data, preproc_Train_Data,preproc_Val_Data,
            embedding_info, input_dim, output_dim)


@pytest.mark.parametrize("TrainData, selected_cols, ValData, "
                         "expection1, expection2,"
                         "expection3, expection4,"
                         "expection5, expection6,"
                         "expection7, expection8",
[
(
# Test1
# Eingabe
# Trainingsdaten
pd.DataFrame({
    'A': pd.Series([100, 200, 400, 100, 500, np.nan],
    dtype=object),
    'B': pd.Series(['Kappa', 'Alpha', 'Beta', 'Gamma', 'Epsilon', 'Kappa'],
    dtype=pd.CategoricalDtype(categories=['Alpha', 'Beta', 'Gamma', 'Epsilon', 'Kappa', 'was_missing'])),
    }),
# Ausgewählte Spalten für Encodierung
['A'],
# Testdaten
pd.DataFrame({
    'A': pd.Series([200, 200, 300, 500, np.nan, 700],dtype=object),
    'B': pd.Series(['Alpha', 'Beta', 'Epsilon', 'Gamma', 'Gamma', 'Beta'],
    dtype=pd.CategoricalDtype(categories=['Alpha', 'Beta', 'Gamma', 'Epsilon', 'was_missing'])),
    }),
# Erwartungen
# Transformation der Trainingsdaten für Embedding (nach OrdinalEncoder)
pd.DataFrame({
    'A': pd.Series([2, 3, 4, 2, 5, 1],dtype='int64'),
    'B': pd.Series(['Kappa', 'Alpha', 'Beta', 'Gamma', 'Epsilon', 'Kappa'],
    dtype=pd.CategoricalDtype(categories=['Alpha', 'Beta', 'Gamma', 'Epsilon', 'Kappa', 'was_missing'])),
    }),
# Transformation der Testdaten für Embedding (nach OrdinalEncoder)
pd.DataFrame({
    'A': pd.Series([3, 3, 0, 5, 1, 0],dtype='int64'),
    'B': pd.Series(['Alpha', 'Beta', 'Epsilon', 'Gamma', 'Gamma', 'Beta'],
    dtype=pd.CategoricalDtype(categories=['Alpha', 'Beta', 'Gamma', 'Epsilon', 'was_missing'])),
}),
# Transformation der Trainingsdaten nach Embedding
pd.DataFrame({
    'B': pd.Series(['Kappa', 'Alpha', 'Beta', 'Gamma', 'Epsilon', 'Kappa'],
    dtype=pd.CategoricalDtype(categories=['Alpha', 'Beta', 'Gamma', 'Epsilon', 'Kappa', 'was_missing'])),
    'A_0': pd.Series([1, 1, 0, 1, 2, 0],dtype='int64'),
    'A_1': pd.Series([0, 1, 2, 0, 0, 1],dtype='int64'),
    }),
# Transformation der Testdaten nach Embedding
pd.DataFrame({
    'B': pd.Series(['Alpha', 'Beta', 'Epsilon', 'Gamma', 'Gamma', 'Beta'],
    dtype=pd.CategoricalDtype(categories=['Alpha', 'Beta', 'Gamma', 'Epsilon', 'was_missing'])),
    'A_0': pd.Series([1, 1, 0, 2, 0, 0],dtype='int64'),
    'A_1': pd.Series([1, 1, 0, 0, 1, 0],dtype='int64'),
}),
# Kategorie-Werte aus den Trainingsdaten erzeugt (categoryvalues)
['unknown','was_missing','100','200','400','500'],
# Encodier-Map des OrdinalEncoder
['unknown','was_missing','100','200','400','500'],
# Zeilenanzahl
6,
# Spaltenanzahl
2
)
])
def test_prep_for_embedding3(TrainData, selected_cols, ValData,
                            expection1, expection2,expection3,
                            expection4,expection5, expection6,
                            expection7, expection8):
    (encoded_Train_Data, encoded_Val_Data, preproc_Train_Data, preproc_Val_Data,
     embedding_info, input_dim, output_dim)=(prep_for_embedding(TrainData, selected_cols, ValData))
    # Tests für korrektes Ordinal Encoding
    pd.testing.assert_frame_equal(encoded_Train_Data, expection1)
    pd.testing.assert_frame_equal(encoded_Val_Data, expection2)
    # Test für korrektes Encoding für Embedding
    pd.testing.assert_frame_equal(preproc_Train_Data, expection3)
    pd.testing.assert_frame_equal(preproc_Val_Data, expection4)
    # Test für korrekte Kategorien der Trainingsdaten
    assert list(embedding_info['A'][0]) == expection5
    # Test für korrekte Encodingmap
    assert list(embedding_info['A'][1].categories_[0]) == expection6
    # Tests der korrekten Berechnung der Dimension der Embedding Matrix
    assert input_dim['A'] == expection7
    assert output_dim['A'] == expection8

