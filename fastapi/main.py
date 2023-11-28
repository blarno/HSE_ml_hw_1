from fastapi import FastAPI, UploadFile
from pydantic import BaseModel
from typing import List
from fastapi.responses import FileResponse
import joblib
import csv
import codecs
import numpy as np
import pandas as pd

app = FastAPI()

class Item(BaseModel):
    name: str
    year: int
    selling_price: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str
    engine: str
    max_power: str
    torque: str
    seats: float

class Items(BaseModel):
    objects: List[Item]

def feature_engineering(df_test):

    for col in ['mileage', 'engine', 'max_power']:
        df_test[col] = df_test[col].str.extract('(\d+)', expand=False).astype(float)

    df_test = df_test.drop('torque', axis = 1)

    for col in ['engine', 'seats']:
        df_test[col] = df_test[col].astype(int)

    df_test['year_sqr'] = df_test['year'] * df_test['year']
    df_test['hp_by_volume'] = df_test['max_power'] / df_test['engine']

    X_test = df_test.drop(['name','selling_price'], axis = 1)

    return X_test


@app.post("/predict_item")
def predict_item(item: Item)-> float:
    model = joblib.load('C:/Users/vladc/PycharmProjects/pythonProject/venv/model.pkl')
    scaler = joblib.load('C:/Users/vladc/PycharmProjects/pythonProject/venv/scaler.pkl')
    encoder = joblib.load('C:/Users/vladc/PycharmProjects/pythonProject/venv/encoder.pkl')
    
    X_test = feature_engineering(pd.DataFrame.from_dict([item.dict()]))
    cols = encoder.get_feature_names_out()
    encoded = encoder.transform(X_test[['fuel', 'seller_type', 'transmission', 'owner', 'seats']])
    encoded = pd.DataFrame(encoded.toarray(), index=X_test.index, columns=cols)
    
    X_test = X_test.drop(['fuel', 'seller_type', 'transmission', 'owner', 'seats'], axis=1)
    X_test = pd.concat([X_test, encoded], axis='columns')
    X_test = pd.DataFrame(scaler.transform(X_test))
    
    return model.predict(X_test)

@app.post("/predict_items")
def predict_items_csv(file: UploadFile):
    model = joblib.load('C:/Users/vladc/PycharmProjects/pythonProject/venv/model.pkl')
    scaler = joblib.load('C:/Users/vladc/PycharmProjects/pythonProject/venv/scaler.pkl')
    encoder = joblib.load('C:/Users/vladc/PycharmProjects/pythonProject/venv/encoder.pkl')
    
    X_test = feature_engineering(pd.read_csv(file.file, sep =',', index_col = [0]))
    cols = encoder.get_feature_names_out()
    encoded = encoder.transform(X_test[['fuel', 'seller_type', 'transmission', 'owner', 'seats']])
    encoded = pd.DataFrame(encoded.toarray(), index=X_test.index, columns=cols)
    
    X_test = X_test.drop(['fuel', 'seller_type', 'transmission', 'owner', 'seats'], axis=1)
    X_test = pd.concat([X_test, encoded], axis='columns')
    X_test = pd.DataFrame(scaler.transform(X_test))
    
    X_test['preds'] = model.predict(X_test)
    X_test.to_csv('export.csv')
    
    return FileResponse('export.csv')
