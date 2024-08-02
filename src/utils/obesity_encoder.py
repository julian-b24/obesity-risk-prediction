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


def get_class_encoder(encoder: dict | LabelEncoder, encoded_class: int) -> str:
    if isinstance(encoder, dict):
        return list(encoder.keys())[encoded_class]