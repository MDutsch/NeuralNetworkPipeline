import pandas as pd
from pandas.api.types import (is_numeric_dtype)
import sklearn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif


def roughly_mutual_information(TrainData, Target, classification,seed):
    TrainDataCopy = TrainData.copy()
    # Zur Berechnung von mi bei category-Spalten, müssen diese Label-encoded werden
    # Label-Encoding der category-Spalten
    label_encoder = sklearn.preprocessing.LabelEncoder()
    cat_features = TrainDataCopy.select_dtypes(include=['category'])
    for col in cat_features.columns:
        if is_numeric_dtype(TrainDataCopy[col].cat.categories):
            # numerische category-Spalten dürfen nicht encoded werden!
            continue
        else:
            # Für deterministische Reihenfolge beim label_encoder muss Wertemenge sortiert werden!
            categoryset = sorted(TrainDataCopy[col].dropna().unique())
            label_encoder.fit(categoryset)
            encoded_feature = label_encoder.transform(TrainDataCopy[col])
            TrainDataCopy[col] = encoded_feature
    # Berechnung Mutual Information
    ''' discrete1 = [pd.api.types.is_integer_dtype(dtype) or 
                   pd.api.types.is_bool_dtype(dtype) or 
                   isinstance(dtype, pd.CategoricalDtype) 
                   for dtype in TrainDataCopy.dtypes] '''
    discrete2 = TrainDataCopy.dtypes.isin(['int64', 'category', 'bool']).values
    if classification:
        mi = mutual_info_classif(TrainDataCopy, Target, discrete_features=discrete2, random_state=seed, n_jobs=1)
    else:
        mi = mutual_info_regression(TrainDataCopy, Target, discrete_features=discrete2, random_state=seed, n_jobs=1)
    mi = pd.Series(mi, name="MI Scores", index=TrainDataCopy.columns)
    # Mutual Information barplot
    mi = mi.sort_values(ascending=True)
    width = np.arange(len(mi))
    ticks = list(mi.index)
    fig, ax = plt.subplots()
    ax.barh(width, mi)
    ax.set_yticks(width)
    ax.set_yticklabels(ticks)
    ax.set_title("Mutual Information Scores")
    fig.tight_layout()
    # Zeige fig im aufrufenden Script dann mit plt.show() an
    return fig





# Anzeige-Funktionen
def show_cardinality(TrainData, TestData, Colnames, maxcard):
    comp_data = pd.concat([TrainData, TestData], ignore_index=True)
    for col in Colnames:
        set = comp_data[col].dropna().unique()
        card = comp_data[col].nunique()
        print('Spalte ' + str(col) + ' besitzt ' + str(card) + ' unterschiedliche Werte')
        if card <= maxcard:
            print('welche ' + str(set) + ' lauten.')


def show_set(Series):
    print("Spalte " + str(Series.name) + " besitzt folgende Werte: ", sorted(pd.unique(Series).tolist()))
