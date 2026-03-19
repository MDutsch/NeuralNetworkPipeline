import numpy as np
import pandas as pd
from pandas.api.types import is_float_dtype, is_integer_dtype, is_numeric_dtype
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler, MinMaxScaler
#-----------------------------------------------------------------------------------------------------------------------
# Hilfsfunktionen
def detect_bool(col):
    bool_set = {0, 1, True, False, 'Ja', 'ja', 'Nein', 'nein', 'True', 'true', 'False', 'false'}
    booldetected = set(col.dropna().unique()).issubset(bool_set)
    return booldetected

def convert_bool(col):
    mapping = {True: 1, 'True': 1, 'true': 1, 'Ja': 1, 'ja': 1,
               False: 0, 'False': 0, 'false': 0, 'Nein': 0, 'nein': 0}
    converted_col = col.map(mapping)
    return converted_col
#-----------------------------------------------------------------------------------------------------------------------
# Prozess-Funktionen
def preprocess_train(TrainData, maxcardhiddencat, maxcardcat, maxmisspercent,
                     detection_excep, drop_excep, impute_excep,
                     cols_for_manual_drop, cols_for_onehot,cols_for_embedding,
                     cols_and_scaler):

    column_info=column_detection_universal(TrainData, maxcardhiddencat,detection_excep)
    TrainData, columns_to_drop = drop_columns_train(TrainData, column_info, maxcardcat, maxmisspercent, drop_excep)
    TrainData, manual_columns_to_drop = manual_columns_drop_train(TrainData, cols_for_manual_drop)
    TrainData, impute_and_convert_info = impute_columns_train(TrainData, column_info, impute_excep)
    TrainData, one_hot_encode_info = one_hot_encode_train(TrainData,cols_for_onehot)
    TrainData, embedding_info, input_dims, output_dims = prep_for_embedding_train(TrainData, cols_for_embedding)
    TrainData, columnscalers_info = scale_numerics_train(TrainData, cols_and_scaler)
    TrainData = TrainData.sort_index(axis=1)
    return (TrainData, columns_to_drop, manual_columns_to_drop, impute_and_convert_info,
            one_hot_encode_info, embedding_info, input_dims, output_dims, columnscalers_info)

def preprocess_testval(Data, columns_to_drop, manual_columns_to_drop,
                       impute_and_convert_info, one_hot_encode_info, embedding_info, columnscalers_info):

    Data=drop_columns_testval(Data, columns_to_drop)
    Data = manual_columns_drop_testval(Data, manual_columns_to_drop)
    Data=impute_columns_testval(Data, impute_and_convert_info)
    Data=one_hot_encode_testval(Data, one_hot_encode_info)
    Data=prep_for_embedding_testval(Data, embedding_info)
    Data=scale_numerics_testval(Data, columnscalers_info)
    Data=Data.sort_index(axis=1)
    return Data
#-----------------------------------------------------------------------------------------------------------------------
# Funktionen für die einzelnen Vorverarbeitungsschritte
def column_detection_universal(Data, maxcardhiddencat, excep):
    # Eckdaten Erfassung in Features, Erfassung der vorgesehenen Datentypen in den Spalten
    info = {}
    for col in Data.columns:
        if col not in excep:
            OrigDatTyp = str(Data[col].dtype)
            card = Data[col].nunique()
            rows = len(Data[col])
            missperc = (100 / rows) * Data[col].isna().sum()
            if not is_numeric_dtype(Data[col]):
                # check non-numeric-columns
                if detect_bool(Data[col]):
                    # hiddenbool in non-numeric?
                    DetecDatType = 'bool'
                else:
                    # its hiddencategory in non-numeric!
                    DetecDatType = 'category'
            else:
                # check numeric columns
                if detect_bool(Data[col]):
                    # hiddenbool in numeric?
                    DetecDatType = 'bool'
                elif (card <= maxcardhiddencat) and (card <= rows * 0.8):
                    # hiddencat in numeric?
                    DetecDatType = 'category'
                elif (is_integer_dtype(Data[col]) or Data[col].dropna().apply(lambda x: x.is_integer()).all()):
                    # int or hiddenint in numeric?
                    DetecDatType = 'int'
                elif is_float_dtype(Data[col]) == True:
                    # float in numeric?
                    DetecDatType = 'float'
                else:
                    # else its non cat-, bool-, int-, float-datatype
                    DetecDatType = "other"
            info[col] = {
                "OriginalDType": OrigDatTyp,
                "DetectedDType": DetecDatType,
                "Cardinality": Data[col].nunique(),
                "MissPercentage": missperc,
            }
        else:
            continue
    column_info = pd.DataFrame(info)
    return column_info

def drop_columns_train(TrainData, column_info, maxcardcat, maxmisspercent, excep):
    # Automatisierte Entfernung von Spalten mit zu hohen Prozentsatz an fehlenden Werten sowie Entfernung
    # von category-Spalten mit zu hoher Kardinalität
    columns_to_drop = []
    for col in TrainData.columns:
        detectedtyp = column_info.loc['DetectedDType', col]
        missperc = column_info.loc['MissPercentage', col]
        card = column_info.loc['Cardinality', col]
        if col not in excep:
            if maxmisspercent < missperc:
                TrainData.drop([col], axis=1, inplace=True)
                columns_to_drop.append(col)
            elif detectedtyp == 'category' and (maxcardcat < card):
                TrainData.drop([col], axis=1, inplace=True)
                columns_to_drop.append(col)
            else:
                continue
        else:
            continue
    return TrainData, columns_to_drop

def drop_columns_testval(Data, columns_to_drop):
    for col in columns_to_drop:
        Data.drop([col], axis=1, inplace=True)
    return Data

def manual_columns_drop_train(TrainData, cols_for_manual_drop):
    manual_columns_to_drop = []
    for col in cols_for_manual_drop:
        TrainData.drop([col], axis=1, inplace=True)
        manual_columns_to_drop.append(col)
    return TrainData, manual_columns_to_drop

def manual_columns_drop_testval(Data, manual_columns_to_drop):
    for col in manual_columns_to_drop:
        Data.drop([col], axis=1, inplace=True)
    return Data

def impute_columns_train(TrainData, column_info, excep):
    # Ergänzung der fehlenden Werte in den Spalten,
    # Umwandlung der Spalten-Datentypen in den vorgesehenen Datentyp,
    # Einführung von Indikatorspalte für fehlende Werte in Spalten
    impute_and_convert_info= {}
    for col in TrainData.columns:
        detectedtyp = column_info.loc['DetectedDType', col]
        missperc = column_info.loc['MissPercentage', col]
        if col not in excep:
            indicator_col = "no"
            # Berechnung der imputierten Werte erfolgt für joblib-Datei immer
            match detectedtyp:
                case 'category':  # Imputieren mit "was_missing" als Wert
                    TrainData[col] = TrainData[col].astype('category')
                    impute_value="was_missing"
                    TrainData[col] = TrainData[col].cat.add_categories([impute_value])
                    if missperc > 0:
                        TrainData[f'{col}_was_missing'] = TrainData[col].isnull().astype('int64')
                        TrainData[col] = TrainData[col].fillna('was_missing')
                        indicator_col = "yes"
                    impute_and_convert_info[col]=[impute_value,indicator_col, detectedtyp, 'category']
                case 'bool':  # Imputieren mit häufigsten Wert mit mode()
                    TrainData[col] = convert_bool(TrainData[col])
                    impute_value=TrainData[col].mode()[0]
                    if missperc > 0:
                        TrainData[f'{col}_was_missing'] = TrainData[col].isnull().astype('int64')
                        TrainData[col] = TrainData[col].fillna(TrainData[col].mode()[0])
                        indicator_col = "yes"
                    TrainData[col] = TrainData[col].astype('int64')
                    impute_and_convert_info[col]=[impute_value,indicator_col, detectedtyp, 'int64']
                case 'int':  # Imputieren mit gerundeten Mittelwert
                    impute_value =TrainData[col].mean()
                    if missperc > 0:
                        TrainData[f'{col}_was_missing'] = TrainData[col].isnull().astype('int64')
                        TrainData[col] = TrainData[col].fillna(impute_value).round().astype('int64')
                        indicator_col = "yes"
                    TrainData[col] = TrainData[col].astype('int64')
                    impute_and_convert_info[col]=[impute_value, indicator_col, detectedtyp, 'int64']
                case 'float':  # Imputieren mit Mittelwert
                    impute_value=TrainData[col].mean()
                    if missperc > 0:
                        TrainData[f'{col}_was_missing'] = TrainData[col].isnull().astype('int64')
                        TrainData[col] = TrainData[col].fillna(impute_value)
                        indicator_col = "yes"
                    TrainData[col] = TrainData[col].astype('float64')
                    impute_and_convert_info[col]=[impute_value, indicator_col, detectedtyp, 'float64']
                case 'other':
                    continue
        else:
            continue
    return TrainData, impute_and_convert_info

def impute_columns_testval(Data, impute_and_convert_info):
    for col in impute_and_convert_info:
        impute_value = impute_and_convert_info[col][0]
        indicator_col=impute_and_convert_info[col][1]
        detectedtyp=impute_and_convert_info[col][2]
        match detectedtyp:
            case 'category':
                Data[col] = Data[col].astype('category')
                Data[col] = Data[col].cat.add_categories([impute_value])
                if indicator_col == 'yes':
                    Data[f'{col}_was_missing'] = Data[col].isnull().astype('int64')
                Data[col] = Data[col].fillna(impute_value)
            case 'bool':
                Data[col] = convert_bool(Data[col])
                if indicator_col == 'yes':
                    Data[f'{col}_was_missing'] = Data[col].isnull().astype('int64')
                Data[col] = Data[col].fillna(impute_value)
                Data[col] = Data[col].astype('int64')
            case 'int':
                if indicator_col == 'yes':
                    Data[f'{col}_was_missing'] = Data[col].isnull().astype('int64')
                Data[col] = Data[col].fillna(impute_value)
                Data[col] = Data[col].astype('int64')
            case 'float':
                if indicator_col == 'yes':
                    Data[f'{col}_was_missing'] = Data[col].isnull().astype('int64')
                Data[col] = Data[col].fillna(impute_value)
                Data[col] = Data[col].astype('float64')
            case 'other':
                continue
    return Data

def one_hot_encode_train(TrainData,columns):
    one_hot_encode_info={}
    for col in columns:
        categoryvalues = set(TrainData[col].unique().astype(str))
        categoryvalues.discard('was_missing')
        categoryvalues = sorted(list(categoryvalues))
        categoryvalues = np.array(['unknown', 'was_missing'] + categoryvalues).astype(str)
        TrainData[col] = TrainData[col].astype(str).astype(pd.CategoricalDtype(categories=categoryvalues))
        Encoder = OneHotEncoder(categories=[categoryvalues], sparse_output=False, dtype='int64')
        OneHotData = Encoder.fit_transform(TrainData[[col]])
        OneHotColsNames = Encoder.get_feature_names_out([col])
        OneHotCols = pd.DataFrame(OneHotData, columns=OneHotColsNames, index=TrainData.index)
        OneHotCols.drop(columns=[f'{col}_was_missing'],inplace=True)
        TrainData.drop(columns=[col],inplace=True)
        TrainData = pd.concat([TrainData, OneHotCols], axis=1)
        one_hot_encode_info[col] = [categoryvalues, Encoder]
    return TrainData, one_hot_encode_info

def one_hot_encode_testval(Data, one_hot_encode_info):
    for col in one_hot_encode_info:
        Data[col] = Data[col].astype(str).astype(pd.CategoricalDtype(categories=one_hot_encode_info[col][0]))
        Data[col] = Data[col].fillna('unknown')
        OneHotData = one_hot_encode_info[col][1].fit_transform(Data[[col]].astype(str))
        OneHotColsNames = one_hot_encode_info[col][1].get_feature_names_out([col])
        OneHotCols = pd.DataFrame(OneHotData, columns=OneHotColsNames, index=Data.index)
        OneHotCols.drop(columns=[f'{col}_was_missing'], inplace=True)
        Data.drop(columns=[col], inplace=True)
        Data = pd.concat([Data, OneHotCols], axis=1)
    return Data

def prep_for_embedding_train(TrainData,columns):
    embedding_info={}
    input_dims={}
    output_dims={}
    for col in columns:
        # Kategorie-Werte aus Feature Einträgen generieren
        categoryvalues = set(TrainData[col].unique().astype(str))
        categoryvalues.discard('was_missing')
        categoryvalues = sorted(list(categoryvalues))
        # Indikator-Kategorie-Werte hinzufügen
        categoryvalues = np.array(['unknown', 'was_missing'] + categoryvalues).astype(str)
        # generierte Kategorie-Werte Feature übergeben
        TrainData[col] = TrainData[col].astype(str).astype(pd.CategoricalDtype(categories=categoryvalues))
        # Encodierung
        Encoder = OrdinalEncoder(categories=[categoryvalues])
        TrainData[col] = Encoder.fit_transform(TrainData[[col]]).ravel().astype('int64')
        # Dimension der Embedding Matrix berechnen
        input_dim = len(categoryvalues)  # Zeilen-Anzahl
        output_dim = min(50, round(len(categoryvalues) ** 0.25))  # Spalten-Anzahl
        embedding_info[col] = [categoryvalues, Encoder]
        input_dims[col] = input_dim
        output_dims[col] = output_dim
    return TrainData, embedding_info, input_dims, output_dims

def prep_for_embedding_testval(Data, embedding_info):
    for col in embedding_info:
        # Unbekannte Kategorie-Werte nach Kategorie-Wertemenge des Trainingsdatensatz transformieren
        Data[col] = Data[col].astype(str).astype(pd.CategoricalDtype(categories=embedding_info[col][0]))
        Data[col] = Data[col].fillna('unknown')
        Data[col] = embedding_info[col][1].transform(Data[[col]].astype(str)).ravel().astype('int64')
    return Data

def scale_numerics_train(TrainData, cols_and_scalers):
    columnscalers_info = {}
    for col in cols_and_scalers:
        if cols_and_scalers[col] == "MinMax":
            Scaler = MinMaxScaler()
        elif cols_and_scalers[col] == "Standard":
            Scaler = StandardScaler()
        else:
            Scaler = StandardScaler()
        Scaler.fit(TrainData[col].values.reshape(-1, 1))
        TrainData[col] = Scaler.transform(TrainData[col].values.reshape(-1, 1))
        columnscalers_info[col] = Scaler
    return TrainData, columnscalers_info

def scale_numerics_testval(Data, columnscalers_info):
    for col in  columnscalers_info:
        Scaler=columnscalers_info[col]
        Data[col]=Scaler.transform(Data[col].values.reshape(-1, 1))
    return Data

