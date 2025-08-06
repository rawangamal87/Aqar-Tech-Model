from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import re

# Load model, vectorizer, encoder
model = joblib.load("xgb_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")
label_encoder = joblib.load("label_encoder.pkl")

app = FastAPI()

# Request structure
class MessageInput(BaseModel):
    message: str

# Emoji removal
def remove_emojis(text):
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"
        "\U0001F300-\U0001F5FF"
        "\U0001F680-\U0001F6FF"
        "\U0001F1E0-\U0001F1FF"
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "\U0001F900-\U0001F9FF"
        "\U0001FA70-\U0001FAFF"
        "\U00002500-\U00002BEF"
        "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

# Cleaning
def clean_message_for_model(message):
    message = remove_emojis(message)
    message = re.sub(r'(?:\+966|00966|0)5\d{8}', '[PHONE]', message)
    message = re.sub(r'(\d{1,3}(?:[.,]?\d{3})+|\d+)\s*(ألف|الف|مليون|دولار|جنيه|ج|ريال|ر\.س)?', '[PRICE]', message)
    message = re.sub(r'[\u064B-\u0652]', '', message)
    message = re.sub(r'[^\w\s]', '', message)
    message = re.sub(r'(.)\1{2,}', r'\1', message)
    return message

# Route
@app.post("/predict/")
def predict_label(data: MessageInput):
    cleaned = clean_message_for_model(data.message)
    vec = vectorizer.transform([cleaned])
    pred = model.predict(vec)
    label = label_encoder.inverse_transform(pred)[0]
    return {"prediction": label}
