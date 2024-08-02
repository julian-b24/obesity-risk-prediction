from sklearn.preprocessing import LabelEncoder

ENCODER_NOBESITY = {
    'Insufficient_Weight': 0,
    'Normal_Weight': 1,
    'Overweight_Level_I': 2,
    'Overweight_Level_II': 3,
    'Obesity_Type_I': 4,
    'Obesity_Type_II': 5,
    'Obesity_Type_III': 6,
}


NCP_ENCODER = {
    'Between 1 and 2': 0.0, 
    'Three': 1.0, 
    'More than 3': 2.0,
    'No answer': 3.0
}

CH20_ENCODER = {
    'Less than a liter': 0, 
    'Between 1L and 2L': 1, 
    'More than 2L': 2
}

FCVC_ENCODER = {
    'Never': 1.0,
    'Sometimes': 2.0,
    'Always': 3.0
}

TUE_ENCODER = {
    'None': 1, 
    'Less than an hour': 2, 
    'Between one and three hours': 3, 
    'More than three hours': 4
}

FAF_ENCODER = {
    'Never': 0, 
    'Twice a week': 1, 
    'Two or three times per week': 2, 
    'Four or more times a week': 3
}


def get_class_encoder(encoder: dict | LabelEncoder, encoded_class: int) -> str:
    if isinstance(encoder, dict):
        return list(encoder.keys())[encoded_class]