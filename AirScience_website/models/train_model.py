import pandas as pd
import pickle
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

# Load dữ liệu
Bamboo = pd.read_csv("models\data\Bamboo_Airways.csv")
Vietnam = pd.read_csv("models\data\Vietnam_Airlines.csv")
Vietjet = pd.read_csv("models\data\VietJet_Air.csv")
Vietravel = pd.read_csv("models\data\Vietravel_Airlines.csv")

# Dự đoán giá vé theo ngày
def train(df, brand_name):
    X = df.drop(columns=["price", "brand", "id"])
    y = df["price"]

    X = pd.get_dummies(X, columns=["destination"])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = GradientBoostingRegressor()
    model.fit(X_scaled, y)

    with open(f"models/model/{brand_name}_model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open(f"models/scaler/{brand_name}_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open(f"models/columns/{brand_name}_columns.pkl", "wb") as f:
        pickle.dump(list(X.columns), f)

# Train và lưu cho từng hãng bay
train(Bamboo, "Bamboo")
train(Vietnam, "Vietnam")
train(Vietjet, "Vietjet")
train(Vietravel, "Vietravel")