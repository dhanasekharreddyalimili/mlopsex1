import joblib
import numpy as np

def init():
    global model
    model = joblib.load("model.pkl")

def run(data):
    inputs = np.array(data['inputs'])
    prediction = model.predict(inputs)
    return prediction.tolist()
