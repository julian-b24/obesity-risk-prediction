import pickle
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder


STRING_COLUMNS = ['Gender', 'CAEC', 'CALC', 'MTRANS']
BOOLEAN_COLUMNS = ['family_history_with_overweight', 'FAVC', 'SMOKE', 'SCC']
DROPED_FEATURES = ['Gender', 'NCP', 'SMOKE', 'MTRANS']


def build_features(df: pd.DataFrame, encoder_path: str = None, pca_path: str = None, deploy: bool = False) -> pd.DataFrame:

    df[BOOLEAN_COLUMNS] = df[BOOLEAN_COLUMNS].replace('yes', 1) 
    df[BOOLEAN_COLUMNS] = df[BOOLEAN_COLUMNS].replace('no', 0) 

    encoders = _load_encoders(encoder_path)

    for column in STRING_COLUMNS:
        encoder = encoders[column]
        df[column] = encoder.transform(df[column])

    df = remove_not_correlated_features(df)
    build_bmi_feature(df)
    std_data = standarize_data(df, df.columns)
    
    pca = _load_pca(pca_path)
    if deploy:
        pca_components = pca.transform(std_data)
    else:
        pca_components = pca.fit_transform(std_data)

    columns = [f'PCA {i}' for i in range(1, pca.n_components_ + 1)]
    components_df = pd.DataFrame(
        data = pca_components,
        columns = columns
    )

    new_df = pd.DataFrame(
        data = std_data,
        columns = df.columns
    )

    for column in components_df.columns:
        new_df[column] = components_df[column]

    return new_df


def build_bmi_feature(df: pd.DataFrame) -> None:
    df['BMI'] = df['Weight'] / (df['Height'])**2


def standarize_data(df: pd.DataFrame, features_list: list[str]) -> np.ndarray:
    data = df.loc[:, features_list].values
    data = StandardScaler().fit_transform(data)

    return data


def build_pca_components(data: np.ndarray, components: int) -> tuple:
    pca = PCA(n_components = components)
    pca_components = pca.fit_transform(data)

    columns = [f'PCA {i}' for i in range(1, components + 1)]
    components_df = pd.DataFrame(
        data = pca_components,
        columns = columns
    )

    return (components_df, pca)


def remove_not_correlated_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop(DROPED_FEATURES, axis=1)
    return df


def _load_pca(path: str | None= None) -> PCA:
    if not path:
        path = '../../results/red_dimension/'

    with open(path + 'pca.pkl', 'rb') as file:
        pca = pickle.load(file)
    
    return pca


def _load_encoders(path: str | None= None) -> dict[str, LabelEncoder]:
    if not path:
        path = '../../results/encoders/'

    with open(path + 'encoders.pkl', 'rb') as file:
        encoders = pickle.load(file)
    
    return encoders