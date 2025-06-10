import pandas as pd
import pickle
import os

# Tiền xử lý input
def input_preprocessing(data):
   feature_adding = ['start_hour_Afternoon', 'start_hour_EarlyMorning', 'start_hour_Evening',
                     'start_hour_LateNight', 'start_hour_Morning',
                     'end_hour_Afternoon', 'end_hour_EarlyMorning', 'end_hour_Evening',
                     'end_hour_LateNight', 'end_hour_Morning',
                     'hand_luggage_7', 'hand_luggage_10', 'hand_luggage_12',
                     'checked_baggage_0', 'checked_baggage_20', 'checked_baggage_23']
   columns = ['start_hour', 'end_hour', 'hand_luggage', 'checked_baggage']
   for feature in feature_adding:
      data[feature] = False
   for index, row in data.iterrows():
      for col in columns:
         tmp = row[col]
         data.at[index, f'{col}_{tmp}'] = True

   return data.drop(columns=columns)

# Tạo bộ dữ liệu theo ngày cho mô hình
def create_data(X):
   expanded_rows = []

   for _, row in X.iterrows():
      for day in range(row['days_left'] - 1, 0, -1):  # giảm từ days_left về 1
         new_row = row.copy()
         new_row['days_left'] = day
         expanded_rows.append(new_row)
   return pd.DataFrame(expanded_rows).reset_index(drop=True)

# Dự đoán giá vé theo ngày
def predict_price(path, pred_df):
    pred_with_id = create_data(pred_df)
    pred = pred_with_id.drop(columns=['flight_id', 'brand', 'price'])
    pred = pd.get_dummies(pred, columns=['destination'])
   
    columns_path = f"models\\columns\\{path}_columns.pkl"
    model_path = f"models\\model\\{path}_model.pkl"
    scaler_path = f"models\\scaler\\{path}_scaler.pkl"

    with open(columns_path, "rb") as f:
         feature_cols = pickle.load(f)
    pred = pred.reindex(columns=feature_cols, fill_value=0)

    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    scaled = scaler.transform(pred)

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    predicted_price = model.predict(scaled)

    result = pred_with_id.copy()
    result['Predicted Price'] = predicted_price
    return result

# Dự đoán quyết định
def threshold_predicted(data, result):
   data = data[['flight_id', 'price']].copy()
   data['cheap_pred'] = 0  

   threshold_prices = result['Predicted Price']
   n = max(1,int(0.1 * len(threshold_prices)))
   cheap_threshold = threshold_prices.nsmallest(n)
   cheap_threshold = cheap_threshold.mean()
   cheap_threshold = cheap_threshold + 80000  # cho sai số trong khoảng 80.000   
   print(f"threshold: {cheap_threshold}")

   data.loc[data['price'] <= cheap_threshold, 'cheap_pred'] = 1
   print(data)
   return data
