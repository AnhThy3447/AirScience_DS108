import os
import pandas as pd
import re

# date = ['2304', '2404', '2504']
# for d in date:
#     folder_path = os.path.join('DATA', d)
#     files = os.listdir(folder_path)
#     formatted_date = f"{d[:2]}-{d[2:]}-2025"
#     for file in files:
#         file_path = os.path.join(folder_path, file)
#         print(f"Processing file: {file_path}")
#         df = pd.read_csv(file_path)
#         df['crawl_date'] = formatted_date
#         df.to_csv(file_path, index=False)

base_path = 'DATA'  
places = ['HAN']

data_by_place = {place: [] for place in places}

for folder in sorted(os.listdir(base_path)):
    folder_path = os.path.join(base_path, folder)
    if os.path.isdir(folder_path):
        for filename in os.listdir(folder_path):
            for place in places:
                pattern = (f"planetrip_SGN.{place}&", f"Planetrip_SGN.{place}&")
                if filename.startswith(pattern):
                    file_path = os.path.join(folder_path, filename)
                    df = pd.read_csv(file_path)
                    data_by_place[place].append(df)

for place in places:
    if data_by_place[place]:
        merged_df = pd.concat(data_by_place[place], ignore_index=True)
        merged_df.to_csv(f'{place}_merged.csv', index=False)
        print(f'Merged data for {place} saved to {place}_merged.csv')