import pandas as pd
import sklearn
from pandas.api.types import is_numeric_dtype
import numpy as np
from pandas.api.types import is_float_dtype, is_integer_dtype
from sklearn.preprocessing import StandardScaler, MinMaxScaler


# Hilfsfunktionen
def divide_train_test_data(comp_data, TrainData):
    preproc_TrainData = comp_data.iloc[:len(TrainData), :]
    preproc_TestData = comp_data.iloc[len(TrainData):, :]
    preproc_TrainData = preproc_TrainData.reset_index(drop=True)
    preproc_TestData = preproc_TestData.reset_index(drop=True)
    return preproc_TrainData, preproc_TestData


def detect_bool(col):
    bool_set = {0, 1, True, False, 'Ja', 'ja', 'Nein', 'nein', 'True', 'true', 'False', 'false'}
    booldetected = set(col.dropna().unique()).issubset(bool_set)
    return booldetected


def convert_bool(col):
    mapping = {True: 1, 'True': 1, 'true': 1, 'Ja': 1, 'ja': 1,
               False: 0, 'False': 0, 'false': 0, 'Nein': 0, 'nein': 0}
    converted_col = col.map(mapping)
    return converted_col
#--------------------------------------------------------------------------------------------------
# DataPreprocessingFunctions
def column_detection(TrainData, TestData, maxcardhiddencat, excep):
    # Eckdaten Erfassung in Features, Erfassung der vorgesehenen Datentypen in den Spalten
    comp_data = pd.concat([TrainData, TestData], ignore_index=True)
    info = {}
    for col in comp_data.columns:
        if col not in excep:
            OrigDatTyp = str(comp_data[col].dtype)
            card = comp_data[col].nunique()
            rows = len(comp_data[col])
            missperc = (100 / rows) * comp_data[col].isna().sum()
            if not is_numeric_dtype(comp_data[col]):
                # check non-numeric-columns
                if detect_bool(comp_data[col]):
                    # hiddenbool in non-numeric?
                    DetecDatType = 'bool'
                else:
                    # its hiddencategory in non-numeric!
                    DetecDatType = 'category'
            else:
                # check numeric columns
                if detect_bool(comp_data[col]):
                    # hiddenbool in numeric?
                    DetecDatType = 'bool'
                elif (card <= maxcardhiddencat) and (card <= rows * 0.8):
                    # hiddencat in numeric?
                    DetecDatType = 'category'
                elif (is_integer_dtype(comp_data[col]) or comp_data[col].dropna().apply(
                        lambda x: x.is_integer()).all()):
                    # int or hiddenint in numeric?
                    DetecDatType = 'int'
                elif is_float_dtype(comp_data[col]) == True:
                    # float in numeric?
                    DetecDatType = 'float'
                else:
                    # else its non cat-, bool-, int-, float-datatype
                    DetecDatType = "other"
            info[col] = {
                "OriginalDType": OrigDatTyp,
                "DetectedDType": DetecDatType,
                "Cardinality": comp_data[col].nunique(),
                "MissPercentage": missperc,
            }
        else:
            continue
    column_info = pd.DataFrame(info)
    return column_info


def drop_columns(TrainData, TestData, column_info, maxcardcat, maxmisspercent, excep):
    # Automatisierte Entfernung von Spalten mit zu hohen Prozentsatz an fehlenden Werten sowie Entfernung
    # von category-Spalten mit zu hoher Kardinalität
    comp_data = pd.concat([TrainData, TestData], ignore_index=True)
    for col in comp_data.columns:
        detectedtyp = column_info.loc['DetectedDType', col]
        missperc = column_info.loc['MissPercentage', col]
        card = column_info.loc['Cardinality', col]
        if col not in excep:
            if maxmisspercent < missperc:
                comp_data.drop([col], axis=1, inplace=True)
            elif detectedtyp == 'category' and (maxcardcat < card):
                comp_data.drop([col], axis=1, inplace=True)
            else:
                continue
        else:
            continue
    preproc_TrainData, preproc_TestData = divide_train_test_data(comp_data, TrainData)
    return preproc_TrainData, preproc_TestData


def impute_columns(TrainData, TestData, column_info, excep):
    # Ergänzung der fehlenden Werte in den Spalten,
    # Umwandlung der Spalten-Datentypen in den vorgesehenen Datentyp,
    # Einführung von Indikatorspalte für fehlende Werte in Spalten
    comp_data = pd.concat([TrainData, TestData], ignore_index=True)
    for col in comp_data.columns:
        detectedtyp = column_info.loc['DetectedDType', col]
        missperc = column_info.loc['MissPercentage', col]
        if col not in excep:
            match detectedtyp:
                case 'category':  # Imputieren mit "was_missing" als Wert
                    if missperc > 0:
                        comp_data[f'{col}_was_missing'] = comp_data[col].isnull().astype('int64')
                        comp_data[col] = comp_data[col].astype('category')
                        comp_data[col] = comp_data[col].cat.add_categories('was_missing')
                        comp_data[col] = comp_data[col].fillna('was_missing')
                    else:
                        comp_data[col] = comp_data[col].astype('category')
                case 'bool':  # Imputieren mit häufigsten Wert
                    if missperc > 0:
                        comp_data[f'{col}_was_missing'] = comp_data[col].isnull().astype('int64')
                        comp_data[col] = convert_bool(comp_data[col])
                        comp_data[col] = comp_data[col].fillna(comp_data[col].mode()[0])
                        comp_data[col] = comp_data[col].astype('int64')
                    else:
                        comp_data[col] = comp_data[col].astype('int64')
                case 'int':  # Imputieren mit gerundeten Mittelwert
                    if missperc > 0:
                        comp_data[f'{col}_was_missing'] = comp_data[col].isnull().astype('int64')
                        comp_data[col] = comp_data[col].fillna(comp_data[col].mean()).round().astype('int64')
                        comp_data[col] = comp_data[col].astype('int64')
                    else:
                        comp_data[col] = comp_data[col].astype('int64')
                case 'float':  # Imputieren mit Mittelwert
                    if missperc > 0:
                        comp_data[f'{col}_was_missing'] = comp_data[col].isnull().astype('int64')
                        comp_data[col] = comp_data[col].fillna(comp_data[col].mean())
                        comp_data[col] = comp_data[col].astype('float64')
                    else:
                        comp_data[col] = comp_data[col].astype('float64')
                case 'other':
                    continue
        else:
            continue
    preproc_TrainData, preproc_TestData = divide_train_test_data(comp_data, TrainData)
    return preproc_TrainData, preproc_TestData


def encode_with_embedding(TrainData, TestData, column):
    comp_data = pd.concat([TrainData, TestData], ignore_index=True)
    # Label Encoding
    label_encoder = sklearn.preprocessing.LabelEncoder()
    categoryset = sorted(comp_data[column].dropna().unique())
    label_encoder.fit(categoryset)
    labelencodedFeature = pd.Series(label_encoder.transform(comp_data[column]))
    # Berechne Kardinalitätszahl
    card_number = comp_data[column].nunique()
    # Berechne Embedding Dimension
    embedding_dim = min(50, round(card_number ** 0.25))
    # Erstelle Embedding Matrix
    embedding_matrix = np.random.normal(0, 1, size=(card_number, embedding_dim))
    cat_embeddings = embedding_matrix[labelencodedFeature]
    embedding_DataFrame = pd.DataFrame(cat_embeddings,
                                       columns=[column + '_' + f'cat_emb_{i}' for i in range(embedding_dim)])
    comp_data.drop([column], axis=1, inplace=True)
    comp_data = pd.concat([comp_data, embedding_DataFrame], ignore_index=False,axis=1)
    preproc_TrainData, preproc_TestData = divide_train_test_data(comp_data, TrainData)
    return preproc_TrainData, preproc_TestData

def one_hot_encode(TrainData, TestData, column):
    comp_data = pd.concat([TrainData, TestData], ignore_index=True)
    DummieCols = pd.get_dummies(comp_data[column])
    for dumcol in DummieCols:
        comp_data[f'{column}_{dumcol}'] = DummieCols[dumcol].astype(int)
    comp_data.drop([column], axis=1, inplace=True)
    preproc_TrainData, preproc_TestData = divide_train_test_data(comp_data, TrainData)
    return preproc_TrainData, preproc_TestData

def scale_columns(TrainData, TestData, MinMaxOrStandard, excep):
    comp_data = pd.concat([TrainData, TestData], ignore_index=True)
    if MinMaxOrStandard == "Standard":
        scaler = StandardScaler()
    else:  # MinMaxScaler ist gewünscht
        scaler = MinMaxScaler()
    # Alle Spalte die numerisch sind, sind float (ehemalig Bool oder Category sind Integer nach den vorherigen Schritten)
    for col in comp_data.select_dtypes(include=["float"]).columns:
        if col in excep:
            continue
        else:
            comp_data[col] = scaler.fit_transform(comp_data[[col]])
    preproc_TrainData, preproc_TestData = divide_train_test_data(comp_data, TrainData)
    return preproc_TrainData, preproc_TestData
