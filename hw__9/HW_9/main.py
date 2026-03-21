import os
import pandas as pd
import streamlit as st
import numpy as np
from src.utils import prepare_data, train_model, read_model
st.set_page_config(
    page_title="Оцени квартиру",
)
model_path = 'lr_fitted.pkl'

total_square = st.sidebar.number_input("Какая площадь квартиры?", 20, 500, 40)

floor = st.sidebar.number_input("Какой этаж?", 1, 65, 1)

# create input DataFrame
inputDF = pd.DataFrame(
    {
        "total_square": total_square,
        "floor": floor,
    },
    index=[0],
)
if not os.path.exists(model_path):
    train_data = prepare_data()
    train_data.to_csv('data.csv')
    train_model(train_data)
model = read_model('lr_fitted.pkl')
preds = model.predict(inputDF).astype(int)

#цена не может опускаться меньше 0, сделаем её не меньше 1 млн
preds = np.maximum(preds, 1000000)

st.image("imgs/realty.jpg")
st.write(f"Оцениваемая стоимость квартиры в рублях: {preds}")