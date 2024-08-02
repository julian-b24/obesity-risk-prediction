import sys
sys.path.append('../../')

import streamlit as st

import pickle
import pandas as pd

from src.streamlit.predict import execute_prediction
from src.utils.obesity_encoder import get_class_encoder, NCP_ENCODER, CH20_ENCODER, FCVC_ENCODER, TUE_ENCODER, FAF_ENCODER
from src.models.models_enums import Models


model = None # Load Model
prediction = None
image_prediction = None

st.title(':warning: Obesity Risk Prediction')
st.markdown('**Welcome to this prediction tool for obesity risk classification!** ')
st.markdown('Here you will enter the information of a person to classity him/her as' +
            ' in a class of the Obesity Scale according to a set of charactetistics.*')

st.markdown('**Rember this is not a medical diagnosis tool, it does not substitue any medical professional.* :male-doctor:')

st.markdown('As a first step, select the model that will be used for the predictions:')
model = st.selectbox(
    'Model',
    [Models.RANDOM_FOREST.name, Models.LOGISTIC_REGRESSION.name, Models.XGBOOST.name],
    key='model'
)

st.markdown('**Pss.. Note:** The best model was the Random Forest (89% Accuracy) *FYI* :eyes:')


with st.form('obesity_form'):
    st.subheader('Basic Information')

    gender = st.radio(
        'Gender',
        ['Male', 'Female'],
        key='gender'
    )

    r1_col1, r1_col2, r1_col3 = st.columns(3)
    with r1_col1:
        age = st.number_input('Age', min_value=1, max_value=120, step=1)

    with r1_col2:
        weight = st.number_input('Weight (kg)', min_value=1.0, max_value=180.0)

    with r1_col3:
        height = st.number_input('Height (m)', min_value=0.1, max_value=2.0)
    

    st.subheader('Dietary and Activity Information')

    r2_col1, r2_col2 = st.columns(2)

    with r2_col1:
        family_history = st.radio(
            'Does someone in your family ever had overweight problems?',
            ['yes', 'no'],
            key='history'
        )
    
    with r2_col2:
        fcvc = st.radio(
            'Frequency of consumption of vegetables',
            ['Never', 'Sometimes', 'Always'],
            key='fcvc'
        )
    

    r3_col1, r3_col2 = st.columns(2)

    with r3_col1:
        favc = st.radio(
            'Eats oftenly high calority foods?',
            ['yes', 'no'],
            key='favc'
        )
    
    with r3_col2:
        ncp = st.selectbox(
            'Number of main meals in a day:',
            ['Between 1 and 2', 'Three', 'More than 3', 'No answer'],
            key='ncp'
        )
    
    r4_col1, r4_col2, r4_col3 = st.columns(3)
    with r4_col1:
        ch2o = st.radio(
            'How much water drinks daily?',
            ['Less than a liter', 'Between 1L and 2L', 'More than 2L'],
            key='ch2o'
        )
    
    with r4_col2:
        scc = st.radio(
            'Keeps track of caloric intake?',
            ['yes', 'no'],
            key='scc'
        )
    
    with r4_col3:
        smoke = st.radio(
            'Smokes?',
            ['yes', 'no'],
            key='smoke'
        )
    
    r5_col1, r5_col2 = st.columns(2)
    with r5_col1:
        caec = st.selectbox(
            'Consumption of food between meals:',
            ['no', 'Sometimes', 'Frequently', 'Always'],
            key='caec'
        )
    
    with r5_col2:
        faf = st.selectbox(
            'Phisical Activity Frequence:',
            ['Never', 'Twice a week', 'Two or three times per week', 'Four or more times a week'],
            key='faf'
        )

    r6_col1, r6_col2 = st.columns(2)

    with r6_col1:
        tue = st.selectbox(
            'Time spent using electronic devices:', 
            ['None', 'Less than an hour', 'Between one and three hours', 'More than three hours'],
            key='tue'
            )
    
    with r6_col2:
        calc = st.selectbox(
            'Alcohol consumption frequence:', 
            ['no', 'Sometimes', 'Frequently', 'Always'],
            key='calc'
            )

    r7_col1, _ = st.columns(2)

    with r7_col1:
        mtrans = st.selectbox(
            'Most used transportation:', 
            ['Automobile', 'Motorbike', 'Bike', 'Public Transportation', 'Walking'],
            key='mtrans'
            )
    

    r8_col1, r8_col2 = st.columns(2)
    st.empty()

    predict = st.form_submit_button('Predict Obesity Risk', type='primary')
    if predict:
        sample = {
            'Gender': gender,
            'Age': age,
            'Height': height,
            'Weight': weight,
            'family_history_with_overweight': family_history,
            'FAVC': favc,
            'FCVC': FCVC_ENCODER[fcvc],
            'CAEC': caec,
            'CH2O': CH20_ENCODER[ch2o],
            'SCC': scc,
            'FAF': FAF_ENCODER[faf],
            'TUE': TUE_ENCODER[tue],
            'CALC': calc,
            'SMOKE': smoke,
            'NCP': NCP_ENCODER[ncp],
            'MTRANS': mtrans.replace(' ', '_')
        }

        prediction = execute_prediction(sample, model_type=model)
        print(sample)
        print(prediction)
        st.write(prediction)
