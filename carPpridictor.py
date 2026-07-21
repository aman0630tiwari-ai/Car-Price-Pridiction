import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression

cars_data = pd.read_csv("Cardetails.csv.xls")

cars_data.dropna(inplace=True)

def model_name(carName):
    return carName.split()[0]

cars_data["name"] = cars_data["name"].apply(model_name)

cars_data.drop(columns=["torque"], inplace=True)

def clean_data(value):
    value = str(value).split(" ")[0]
    value = value.strip()

    if value == "":
        value = 0

    return float(value)

cars_data["mileage"] = cars_data["mileage"].apply(clean_data)
cars_data["engine"] = cars_data["engine"].apply(clean_data)
cars_data["max_power"] = cars_data["max_power"].apply(clean_data)

name_encoder = LabelEncoder()
fuel_encoder = LabelEncoder()
seller_encoder = LabelEncoder()
transmission_encoder = LabelEncoder()
owner_encoder = LabelEncoder()

cars_data["name"] = name_encoder.fit_transform(cars_data["name"])
cars_data["fuel"] = fuel_encoder.fit_transform(cars_data["fuel"])
cars_data["seller_type"] = seller_encoder.fit_transform(cars_data["seller_type"])
cars_data["transmission"] = transmission_encoder.fit_transform(cars_data["transmission"])
cars_data["owner"] = owner_encoder.fit_transform(cars_data["owner"])

X = cars_data.drop("selling_price", axis=1)
Y = cars_data["selling_price"]

x_train, x_test, y_train, y_test = train_test_split(
    X,
    Y,
    test_size=0.20,
    random_state=42
)

model = LinearRegression()
model.fit(x_train, y_train)

accuracy = model.score(x_test, y_test)


st.title("🚗 Car Price Prediction")

st.write("Predict the selling price of a used car.")

brand = st.selectbox(
    "Select Car Brand",
    name_encoder.classes_
)

year = st.slider("Manufacturing Year", 1994, 2024, 2018)

km_driven = st.slider(
    "KM Driven",
    0,
    300000,
    50000
)

fuel = st.selectbox(
    "Fuel Type",
    fuel_encoder.classes_
)

seller = st.selectbox(
    "Seller Type",
    seller_encoder.classes_
)

transmission = st.selectbox(
    "Transmission",
    transmission_encoder.classes_
)

owner = st.selectbox(
    "Owner",
    owner_encoder.classes_
)

mileage = st.slider(
    "Mileage",
    5.0,
    40.0,
    20.0
)

engine = st.slider(
    "Engine (CC)",
    500,
    5000,
    1200
)

max_power = st.slider(
    "Max Power",
    20,
    300,
    80
)

seats = st.slider(
    "Seats",
    2,
    10,
    5
)

if st.button("Predict Price"):

    input_data = pd.DataFrame(
        [[
            name_encoder.transform([brand])[0],
            year,
            km_driven,
            fuel_encoder.transform([fuel])[0],
            seller_encoder.transform([seller])[0],
            transmission_encoder.transform([transmission])[0],
            owner_encoder.transform([owner])[0],
            mileage,
            engine,
            max_power,
            seats
        ]],
        columns=X.columns
    )

    prediction = model.predict(input_data)

    st.success(f"Predicted Selling Price : ₹ {prediction[0]:,.2f}")

st.write("")

st.write(f"### Model Accuracy : {accuracy*100:.2f}%")