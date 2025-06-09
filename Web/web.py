import pandas as pd
import numpy as np
pd.set_option('display.max_columns', 200)

# Bộ dữ liệu train mô hình
Bamboo = pd.read_csv()
Vietnam = pd.read_csv()
Vietjet = pd.read_csv()
Vietravel = pd.read_csv()

# Bộ dữ liệu chuyến bay
"""
   Bao gồm các thuộc tính:
   'brand' : 'Bamboo Airways',
   'price' : '123456',
   'destination' : 'Phú Quốc (PQC)\nSân bay Phú Quốc',
   'hand_luggage' : 7,
   'checked_baggage' : 0,
   'start_hour' : 'Evening',
   'end_hour' : 'Evening',
   'trip_mins' : 60, 
   'is_holiday' : 2,
   'days_left' : 1
"""
data = pd.read_csv()


# Tiền xử lý input
def input_preprocessing(data):
   feature_adding = ['start_hour_Afternoon', 'start_hour_EarlyMorning', 'start_hour_Evening',
                     'start_hour_LateNight', 'start_hour_Morning',
                     'end_hour_Afternoon', 'end_hour_EarlyMorning', 'end_hour_Evening',
                     'end_hour_LateNight', 'end_hour_Morning',
                     'hand_luggage_7', 'hand_luggage_10', 'hand_luggage_12',
                     'checked_baggage_0', 'checked_baggage_20', 'checked_baggage_23']
   columns = ['start_hour', 'end_hour', 'hand_luggage', 'checked_baggage']
   id = 1
   for feature in feature_adding:
      data[feature] = False
   for index, row in data.iterrows():
      data.at[index, 'id'] = id # Thêm id
      id += 1
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

# Modeling
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

# Chuẩn hóa dữ liệu
def scale_data(train, pred):
   scaler = StandardScaler() 
   train = scaler.fit_transform(train)
   pred = scaler.transform(pred)
   train=pd.DataFrame(train)
   pred=pd.DataFrame(pred)
   return train, pred

# Dự đoán giá vé theo ngày
def predict(train, pred):
   pred_with_id = create_data(pred)

   pred = pred_with_id.drop(columns=['id', 'brand', 'price'])

   x_train = train.drop(columns=['price', 'brand', 'id'])
   y_train = train['price']

   pred = pd.get_dummies(pred, columns=['destination'])
   x_train = pd.get_dummies(x_train, columns=['destination'])

   train_scaled, pred_scaled = scale_data(x_train, pred)

   modelGBR = GradientBoostingRegressor()
   modelGBR.fit(train_scaled, y_train)

   predicted_price = modelGBR.predict(pred_scaled)

   result = pred_with_id.copy()
   result['Predicted Price'] = predicted_price

   return result

# Dự đoán quyết định
def threshold_predicted(data, result):
   data = data[['id', 'price']].copy()
   data['cheap_pred'] = 0  

   for id_value in result['id'].unique():
      df = result[(result['id'] == id_value)].copy()

      threshold_prices = df['Predicted Price']
      n = max(1,int(0.1 * len(threshold_prices)))
      cheap_threshold = threshold_prices.nsmallest(n)
      cheap_threshold = cheap_threshold.mean()

      mask = (data['id'] == id_value) & (data['price'] <= cheap_threshold)
      data.loc[mask, 'cheap_pred'] = 1
   return data

if __name__== '__main__':
   data = input_preprocessing(data)
   
   results = []

   for brand in data['brand'].unique():
      df = data[data['brand'] == brand].copy()
      if brand == 'Bamboo Airways':
         result = predict(Bamboo, df)
      elif brand == 'Vietravel Airlines':
         result = predict(Vietravel, df)
      elif brand == 'Vietjet Air':
         result = predict(Vietjet, df)
      elif brand == 'Vietnam Airlines':
         result = predict(Vietnam, df)
      else: 
         continue
      result = threshold_predicted(df, result)
      results.append(result)

   # results có dạng id, price, cheap_pred (cheap_pred là quyết định nên mua hay không)
   results = pd.concat(results, ignore_index=True)
