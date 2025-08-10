from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# تحميل الموديلات والـ preprocessing tools
model_label = joblib.load('model_label.pkl')
model_area = joblib.load('model_area.pkl')
model_city = joblib.load('model_city.pkl')

vectorizer = joblib.load('tfidf_vectorizer.pkl')
label_encoder = joblib.load('label_encoder.pkl')
city_encoder = joblib.load('city_encoder.pkl')
min_max_scaler = joblib.load('min_max_scaler.pkl')

app = FastAPI()

class InputData(BaseModel):
    message: str

@app.post("/predict/")
def predict(data: InputData):
    x_tfidf = vectorizer.transform([data.message])
    
    pred_label_enc = model_label.predict(x_tfidf)[0]
    pred_label = label_encoder.inverse_transform([pred_label_enc])[0]

    pred_area_scaled = model_area.predict(x_tfidf)[0]
    pred_area = min_max_scaler.inverse_transform(np.array(pred_area_scaled).reshape(-1, 1))[0][0]

    pred_city_enc = model_city.predict(x_tfidf)[0]
    pred_city = city_encoder.inverse_transform([pred_city_enc])[0]

    return {
        "predicted_label": pred_label,
        "predicted_area": pred_area,
        "predicted_city": pred_city
    }
