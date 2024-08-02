import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


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