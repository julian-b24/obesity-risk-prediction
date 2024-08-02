import sys
import os
sys.path.append('../../')

import pandas as pd
import pickle

from src.models.models_enums import Models
from src.processing.features_building import build_features
from src.utils.obesity_encoder import get_class_encoder, ENCODER_NOBESITY


RESULTS_PATH = './results/'
MODELS_PATH = RESULTS_PATH + 'models/'


def sample_to_df(sample: dict) -> pd.DataFrame:
    return pd.DataFrame(sample, index=[0])


def execute_prediction(sample: dict, model_type: Models) -> str:
    df = sample_to_df(sample)
    df = pd.concat([df, df])
    df = build_features(df, 'results/encoders/', 'results/red_dimension/')

    model = load_model(model_type)
    prediction_encoded = model.predict(df)
    prediction_class = get_class_encoder(ENCODER_NOBESITY, prediction_encoded[0])

    return prediction_class



def load_model(model_type: Models):
    models = {
        Models.LOGISTIC_REGRESSION.name: Models.LOGISTIC_REGRESSION.value, 
        Models.RANDOM_FOREST.name: Models.RANDOM_FOREST.value, 
        Models.XGBOOST.name: Models.XGBOOST.value, 
    }

    with open(MODELS_PATH + f'model.{models[model_type]}.pkl', 'rb') as file:
        model = pickle.load(file)
    return model