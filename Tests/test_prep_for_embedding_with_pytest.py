# Testbeschreibung:
# -> wird categories für das Feature A und den OrdinalEncoder richtig aus Trainingsdaten
#    generiert (Werte und Reihenfolge)?
# -> ordnet der OrdinalEncoder entsprechend der Reihenfolge in categories die richtigen Ziffern zu?
# -> wird input_dim und output_dim richtig berechnet?
# -> werden Testdaten richtig codiert?
# -> werden Werte in den Testdaten welche nicht in den generierten categories sind mit "unknown" codiert?

# Eigene Bibliotheken
from Bibliotheken.DataPreprocessingFunctions import *
# Allgemeine Bibliothek
import numpy as np
import pandas as pd
import pytest

def assign_embedding_vektor(Data,columns,output_dims):
    embedding_matrix=[[0,0],
                      [0,1],
                      [1,0],
                      [1,1],
                      [0,2],
                      [2,0],
                      [2,2]]
    for col in columns:
        Vektors=[]
        for i in Data[col]:
            Vektors.append(embedding_matrix[i])
        for i in range(0, output_dims[col]):
            Data[col+'_'+str(i)] = pd.Series(np.array(Vektors)[:,i].tolist()).astype('int64')
        Data.drop([col], axis=1, inplace=True)
    return Data


def prep_for_embedding(Train_Data, selected_cols, Val_Data):
    encoded_Train_Data, embedding_info,input_dim, output_dim=prep_for_embedding_train(Train_Data, selected_cols)
    encoded_Val_Data=prep_for_embedding_testval(Val_Data, embedding_info)
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
    'A': pd.Series(['x1', 'x4', 'x5', 'x2', 'x6', 'x4'],
    dtype=pd.CategoricalDtype(categories=['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'was_missing'])),
    'B': pd.Series(['Kappa', 'Alpha', 'Beta', 'Gamma', 'Epsilon', 'Kappa'],
    dtype=pd.CategoricalDtype(categories=['Alpha', 'Beta', 'Gamma', 'Epsilon', 'Kappa', 'was_missing'])),
    }),
# Ausgewählte Spalten für Encodierung
['A'],
# Testdaten
pd.DataFrame({
    'A': pd.Series(['x3', 'x7', 'x5', 'x2', 'x6', 'x8'],
    dtype=pd.CategoricalDtype(categories=['x3', 'x7', 'x2', 'x6','x8', 'x5', 'was_missing'])),
    'B': pd.Series(['Alpha', 'Beta', 'Epsilon', 'Gamma', 'Gamma', 'Beta'],
    dtype=pd.CategoricalDtype(categories=['Alpha', 'Beta', 'Gamma', 'Epsilon', 'was_missing'])),
    }),
# Erwartungen
# Transformation der Trainingsdaten für Embedding (nach OrdinalEncoder)
pd.DataFrame({
    'A': pd.Series([2, 4, 5, 3, 6, 4],dtype='int64'),
    'B': pd.Series(['Kappa', 'Alpha', 'Beta', 'Gamma', 'Epsilon', 'Kappa'],
    dtype=pd.CategoricalDtype(categories=['Alpha', 'Beta', 'Gamma', 'Epsilon', 'Kappa', 'was_missing'])),
    }),
# Transformation der Testdaten für Embedding (nach OrdinalEncoder)
pd.DataFrame({
    'A': pd.Series([0, 0, 5, 3, 6, 0],dtype='int64'),
    'B': pd.Series(['Alpha', 'Beta', 'Epsilon', 'Gamma', 'Gamma', 'Beta'],
    dtype=pd.CategoricalDtype(categories=['Alpha', 'Beta', 'Gamma', 'Epsilon', 'was_missing'])),
}),
# Transformation der Trainingsdaten nach Embedding
pd.DataFrame({
    'B': pd.Series(['Kappa', 'Alpha', 'Beta', 'Gamma', 'Epsilon', 'Kappa'],
    dtype=pd.CategoricalDtype(categories=['Alpha', 'Beta', 'Gamma', 'Epsilon', 'Kappa', 'was_missing'])),
    'A_0': pd.Series([1, 0, 2, 1, 2, 0],dtype='int64'),
    'A_1': pd.Series([0, 2, 0, 1, 2, 2],dtype='int64'),
    }),
# Transformation der Testdaten nach Embedding
pd.DataFrame({
    'B': pd.Series(['Alpha', 'Beta', 'Epsilon', 'Gamma', 'Gamma', 'Beta'],
    dtype=pd.CategoricalDtype(categories=['Alpha', 'Beta', 'Gamma', 'Epsilon', 'was_missing'])),
    'A_0': pd.Series([0, 0, 2, 1, 2, 0],dtype='int64'),
    'A_1': pd.Series([0, 0, 0, 1, 2, 0],dtype='int64'),
}),
# Kategorie-Werte aus den Trainingsdaten erzeugt (categoryvalues)
['unknown','was_missing','x1','x2','x4','x5','x6'],
# Encodier-Map des OrdinalEncoder
['unknown','was_missing','x1','x2','x4','x5','x6'],
# Zeilenanzahl
7,
# Spaltenanzahl
2
)
])
def test_prep_for_embedding2(TrainData, selected_cols, ValData,
                             expection1, expection2, expection3, expection4,
                             expection5, expection6, expection7, expection8):
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

