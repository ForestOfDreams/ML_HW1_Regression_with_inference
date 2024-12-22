from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from typing import List
from fastapi.responses import FileResponse
import joblib
import pandas as pd
import os

app = FastAPI()

pipeline = joblib.load('preprocessor_pipeline_with_model.pkl')

TEMP_DIR = "./temp"
os.makedirs(TEMP_DIR, exist_ok=True)

class CarFeatures(BaseModel):
    name: str
    year: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str
    engine: str
    max_power: str
    seats: int

class CarFeaturesCollection(BaseModel):
    features: List[CarFeatures]

def preprocess_data(input_data: pd.DataFrame) -> pd.DataFrame:
    input_data['brand'] = input_data['name'].apply(lambda x: x.split()[0])
    input_data['model'] = input_data['name'].apply(lambda x: ' '.join(x.split()[1:]))

    input_data['mileage'] = input_data['mileage'].str.replace(' kmpl', '').str.replace(' km/kg', '')
    input_data['mileage'] = pd.to_numeric(input_data['mileage'], errors='coerce')

    input_data['engine'] = input_data['engine'].str.replace(' CC', '')
    input_data['engine'] = pd.to_numeric(input_data['engine'], errors='coerce')

    input_data['max_power'] = input_data['max_power'].str.replace(' bhp', '')
    input_data['max_power'] = pd.to_numeric(input_data['max_power'], errors='coerce')

    input_data = input_data.fillna(0)
    return input_data

@app.post("/predict")
def predict(features: CarFeatures):
    try:
        input_data = pd.DataFrame([features.model_dump()])

        processed_data = preprocess_data(input_data)

        prediction = pipeline.predict(processed_data)
        return {"predicted_price": prediction[0]}
    except Exception as e:
        return {"error": str(e)}

@app.post("/predict_items")
def predict_items(data: CarFeaturesCollection):
    try:
        feature_df = pd.DataFrame([item.model_dump() for item in data.features])

        processed_features = preprocess_data(feature_df)

        predictions = pipeline.predict(processed_features)

        return {"predicted_prices": predictions.tolist()}
    except Exception as e:
        return {"error": str(e)}

@app.post("/predict_from_csv")
async def predict_from_csv(file: UploadFile = File(...)):
    try:
        input_df = pd.read_csv(file.file)

        processed_data = preprocess_data(input_df)

        predictions = pipeline.predict(processed_data)

        input_df['predicted_price'] = predictions

        output_path = os.path.join(TEMP_DIR, "predictions.csv")
        input_df.to_csv(output_path, index=False)

        return FileResponse(output_path, media_type="text/csv", filename="predictions.csv")
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
