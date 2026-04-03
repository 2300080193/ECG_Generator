import pandas as pd
import os

def get_data():
    path = os.path.join(os.path.dirname(__file__), "../data/ecg.csv")
    df = pd.read_csv(path, header=None)

    X = df.iloc[:, :-1].values   # 140 values
    y = df.iloc[:, -1].values    # labels

    return X, y