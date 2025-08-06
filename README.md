# Aqar-Tech-Model Messages Classifier

- Arabic message classification using a trained XGBoost model.
- Preprocessing with TF-IDF and Scikit-learn pipeline.
- FastAPI backend with `/predict/` endpoint.
- Swagger UI for interactive testing.



# Model

- **Model Type**: XGBoost Classifier
- **Vectorizer**: TF-IDF
- **Training Data**: Arabic real estate messages (offers and requests)
- **Output**: Class (`"Offer"` or `"Request"`)


  # Project Structure

|__ main.py                                                                             # Fast api
|__ arabic_real_estate_message_intelligence_system.py                                   # Model(preprocessing & prediction)
|__ xgb_model.pkl                                                                       # Trained ML Model
|__ requirements.txt                                                                    # Python dependencies
