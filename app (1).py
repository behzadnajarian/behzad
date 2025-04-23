
import streamlit as st
import pandas as pd
import numpy as np
import joblib

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# بارگذاری دیتاست
df = pd.read_csv("dataset_shahriar_housing.csv")
X = df.drop(columns=["قیمت نهایی (تومان)"])
y = df["قیمت نهایی (تومان)"]
X_encoded = pd.get_dummies(X, columns=["منطقه", "نوع ملک"])

# آموزش مدل
model = LinearRegression()
model.fit(X_encoded, y)

# رابط کاربری Streamlit
st.title("🧱 پیش‌بینی قیمت خانه در شهریار")
st.write("ویژگی‌های خونه‌ات رو وارد کن:")

area = st.selectbox("منطقه", ['فاز 1', 'فاز 3', 'باغستان', 'کردزار', 'وائین', 'فردوسیه'])
ptype = st.selectbox("نوع ملک", ['آپارتمان', 'ویلایی', 'باغ ویلا'])
size = st.slider("متراژ (متر مربع)", 50, 500, 100)
rooms = st.slider("تعداد اتاق خواب", 1, 6, 2)
year = st.slider("سال ساخت", 1380, 1402, 1395)
parking = st.checkbox("پارکینگ دارد")
elevator = st.checkbox("آسانسور دارد") if ptype == "آپارتمان" else False
storage = st.checkbox("انباری دارد")
pool = st.checkbox("استخر دارد") if ptype != "آپارتمان" else False

# آماده‌سازی ورودی کاربر
input_data = {
    'متراژ': size,
    'تعداد اتاق': rooms,
    'سال ساخت': year,
    'پارکینگ': int(parking),
    'آسانسور': int(elevator),
    'انباری': int(storage),
    'استخر': int(pool),
}

for col in X_encoded.columns:
    if 'منطقه_' in col:
        input_data[col] = 1 if col == f'منطقه_{area}' else 0
    elif 'نوع ملک_' in col:
        input_data[col] = 1 if col == f'نوع ملک_{ptype}' else 0

input_df = pd.DataFrame([input_data])
prediction = model.predict(input_df)[0]

st.success(f"💰 قیمت تخمینی خانه: {int(prediction):,} تومان")
