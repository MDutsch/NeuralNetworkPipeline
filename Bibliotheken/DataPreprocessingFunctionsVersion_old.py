import pandas as pd
from pandas.api.types import is_numeric_dtype, is_bool_dtype
import numpy as np
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
# Pipeline bestehend aus DataPreprocessingFunctions
def simple_pipeline(TrainData,TestData,maxnumcard,maxcatcard, droprate, hiddenint, excep):
    TrainData,TestData=detect_categoricals(TrainData,TestData,maxnumcard,excep)
    TrainData,TestData=drop(TrainData,TestData,droprate,maxcatcard,excep)
    TrainData,TestData=handleMissingValues(TrainData,TestData,hiddenint,excep)
    TrainData,TestData=one_hot_for_cat(TrainData,TestData,excep)
    return TrainData, TestData

# DataPreprocessingFunctions
def detect_categoricals(TrainData, TestData, maxcard, excep):
    comp_data = pd.concat([TrainData, TestData], ignore_index=True)
    # print("Spalten-Datentypen vor Umwandlung: \n" + str(comp_data.dtypes))
    for col in comp_data.columns:
        if not detect_bool(comp_data[col]):  # keine boolesche/versteckte boolesche Spalte
            if col not in excep:
                if not is_numeric_dtype(comp_data[col]):  # keine numerische Spalte
                    # print("nicht-numerische Spalte " + col + ", Wertemenge aus " + str(comp_data[col].nunique()) + " Elementen")  # Testprint
                    comp_data[col] = comp_data[col].astype('category')
                else:  # numerische Spalte!
                    card = comp_data[col].nunique()
                    rows = len(comp_data[col])
                    # print("numerische Spalte " + col + ", Wertemenge aus " + str(card) + " Elementen")  # Testprint
                    if (maxcard >= card) and (rows * 0.8 >= card):
                        comp_data[col] = comp_data[col].astype('category')
                    else:
                        continue
            else:
                continue
        else:
            continue
    # print("Spalten-Datentypen nach Umwandlung: \n" + str(comp_data.dtypes))  # Testprint
    preproc_TrainData, preproc_TestData = divide_train_test_data(comp_data, TrainData)
    return preproc_TrainData, preproc_TestData

def drop(TrainData, TestData, droprate, maxcard, excep):
    comp_data = pd.concat([TrainData, TestData], ignore_index=True)
    for col in comp_data:
        if col not in excep:
            amountofmissing = np.dot(100 / len(comp_data), (comp_data[col].isnull() == True).sum())
            # print("Spalte " + col + " mit " + str(amountofmissing) + " an fehlenden Werten")  # Testprint
            if comp_data[col].dtypes == 'category':  # kategorische Variablen
                card = comp_data[col].nunique()
                # print("Spalte " + col + " ist kategorisch und besitzt " + str(card) + " einzigartige Werte")  # Testprint
                if (card > maxcard) or (amountofmissing > droprate):
                    comp_data.drop([col], axis=1, inplace=True)
                    # print("Spalte " + col + " entfernt")  # Testprint
                else:
                    continue
            else:  # numerische Variable
                # print("Spalte " + col + " ist numerisch")  # Testprint
                if amountofmissing > droprate:
                    comp_data.drop([col], axis=1, inplace=True)
                    # print("Spalte " + col + " entfernt")  # Testprint
                else:
                    continue
        else:
            continue
    preproc_TrainData, preproc_TestData = divide_train_test_data(comp_data, TrainData)
    return preproc_TrainData, preproc_TestData

def handleMissingValues(TrainData, TestData, hiddenint, excep):
    # Imputation für alle Datentypen von Spalten, zusätzlich Extension für numerische Spalten
    comp_data = pd.concat([TrainData, TestData], ignore_index=True)
    for col in comp_data.columns:
        if col not in excep:
            missing_count = (comp_data[col].isnull() == True).sum()
            if comp_data.dtypes[col] == 'category' and missing_count > 0:
                comp_data[col] = comp_data[col].cat.add_categories('missing')
                comp_data[col] = comp_data[col].fillna('missing')
            else:  # Behandlung von bool- und numerischen Spalten
                if is_bool_dtype(comp_data[col]) or detect_bool(comp_data[col]): # boolsche Spalten
                    comp_data[col] = convert_bool(comp_data[col])
                    if missing_count > 0:
                        comp_data[col] = comp_data[col].fillna(comp_data[col].mode()[0])
                    comp_data[col] = comp_data[col].astype(int)
                else:  # numerische Spalten werden hier behandelt
                    if missing_count > 0:
                        # Extension and Imputation
                        comp_data[f'{col}_was_missing'] = comp_data[col].isnull().astype(int)
                        if col in hiddenint:
                            comp_data[col] = comp_data[col].fillna(comp_data[col].mean()).round().astype('int64')
                        else:
                            comp_data[col] = comp_data[col].fillna(comp_data[col].mean())
                    else:
                        if col in hiddenint:
                            comp_data[col] = comp_data[col].astype('int64')

    preproc_TrainData, preproc_TestData = divide_train_test_data(comp_data, TrainData)
    return preproc_TrainData, preproc_TestData


def one_hot_for_cat(TrainData, TestData, excep):
    comp_data = pd.concat([TrainData, TestData], ignore_index=True)
    for col in comp_data:
        if col in excep:
            continue
        else:
            if comp_data.dtypes[col] == "category":
                DummieCols = pd.get_dummies(comp_data[col])
                for dumcol in DummieCols:
                    comp_data[f'{col}_{dumcol}'] = DummieCols[dumcol].astype(int)
                comp_data.drop([col], axis=1, inplace=True)
            else:
                continue
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