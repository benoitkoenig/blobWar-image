import datetime
import pandas as pd

columns = ["datetime", "index", "state_data", "boxes", "classification_logits", "label_boxes"]
file_path = "tracking/training_data.csv"

def reset_data():
    df = pd.DataFrame({}, columns=columns)
    df.to_csv(file_path, header=True, index=False)

def save_data(index, state_data, boxes, classification_logits, label_boxes):
    df = pd.DataFrame({
        "datetime": [datetime.datetime.now()],
        "index": [index],
        "state_data": [state_data],
        "boxes": [boxes],
        "classification_logits": [classification_logits],
        "label_boxes": [label_boxes],
    }, columns=columns)
    df.to_csv(file_path, mode="a", header=False, index=False)

def get_dataframes():
    return pd.read_csv(file_path)
