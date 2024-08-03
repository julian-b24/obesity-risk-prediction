import sys
import os
sys.path.append('../../')

import pandas as pd
import streamlit as st
import pickle

from dotenv import load_dotenv

from google.cloud import storage

from src.models.models_enums import Models
from src.processing.features_building import build_features
from src.utils.obesity_encoder import get_class_encoder, ENCODER_NOBESITY

load_dotenv()

RESULTS_PATH = './results/'
MODELS_PATH = RESULTS_PATH + 'models/'
DEPLOY = True

gcp_credentials = st.secrets["gcp_service_account"]
#gcp_credentials_json = json.dumps(gcp_credentials)
gcp_credentials_dict = dict(gcp_credentials)

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

    model_name = f'model.{models[model_type]}.pkl'

    if DEPLOY:
        client = storage.Client.from_service_account_info(gcp_credentials_dict)
        bucket = client.get_bucket(st.secrets["GCS_BUCKET_NAME"])
        blob = bucket.blob(model_name)
        pickle_in = blob.download_as_string()
        model = pickle.loads(pickle_in)

    else:
        with open(MODELS_PATH + model_name, 'rb') as file:
            model = pickle.load(file)
    return model