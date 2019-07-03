import datetime
import pandas as pd

columns = ["datetime", "label", "logits", "entropy"]
file_path = "tracking/training_data.csv"

def reset_data():
    df = pd.DataFrame({}, columns=columns)
    df.to_csv(file_path, header=True, index=False)

def save_data(label, logits, entropy):
    df = pd.DataFrame({
        "datetime": [datetime.datetime.now()],
        "label": [label],
        "logits": [logits],
        "entropy": [entropy],
    }, columns=columns)
    df.to_csv(file_path, mode="a", header=False, index=False)

def get_dataframes():
    return pd.read_csv(file_path)
