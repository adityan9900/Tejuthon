#%% Packages

import pandas as pd
import numpy as np
from dictionaries import clean_user_country, clean_device, clean_nones, clean_hotel_type, clean_parent_brand_name

#%% Load data

activity_data_original = pd.read_csv("activity_data.csv")
hotel_data_original = pd.read_csv("hotel_data.csv")

#%% Copy and merge data

activity_data = activity_data_original.copy()
hotel_data = hotel_data_original.copy()

merged_data = pd.merge(activity_data, hotel_data)


#%% Convert names

merged_data.replace(clean_user_country, inplace = True)
merged_data.replace(clean_device, inplace = True)
merged_data.replace(clean_nones, inplace = True)
merged_data.replace(clean_hotel_type, inplace = True)
merged_data['parent_brand_name'].fillna('Independent')
merged_data['parent_brand_name'].map(clean_parent_brand_name).fillna("Other")

#%% Delete rows

# device is other (0)
merged_data = merged_data[merged_data['device'] != 'other']

# city_name is Union City
merged_data = merged_data[merged_data['city_name'] != "Union City"]


#%% Encode to discrete

def encode(column_name):
    merged_data[column_name] = merged_data[column_name].astype('category')
    merged_data[column_name + '_enc'] = merged_data[column_name].cat.codes

encode('user_country')
encode('device')
encode('user_action')
encode('city_name')
encode('hotel_type')
encode('brand_name')
encode('parent_brand_name')

#%% Random set of 100,000

np.random.seed(14)
merged_data = merged_data.take(np.random.permutation(len(merged_data))[:500])

#%% Simplify user actions

unique_users = list(set(merged_data['user_id']))

new_merged = pd.DataFrame(columns = ['user_id', 'hotel_id', 'review_count', 'user_action_enc', 'user_count_name_enc'])

#%%
for i in unique_users:

    simple_data = merged_data[merged_data['user_id'] == i]
    simple_min = min(simple_data['user_action_enc'])
    simple_data = simple_data[simple_data['user_action_enc'] == simple_min]
    
    simple_row = pd.DataFrame(columns = ('user_id','hotel_id', 'review_count', 'user_action_enc', 'user_country_enc', 'device_enc', 'city_name_enc','hotel_type_enc', 'brand_name_enc', 'parent_brand_name_enc'))

    simple_row = simple_data.iloc[:1,]
    
    new_merged = new_merged.append(simple_row, ignore_index = True)

#%% Add hotel popularity

review_data = hotel_data_original.copy()
q1 = review_data.review_count.quantile([0.25,0.5,0.75])[0.25]
q2 = review_data.review_count.quantile([0.25,0.5,0.75])[0.5]
q3 = review_data.review_count.quantile([0.25,0.5,0.75])[0.75]

review_data.loc[:, "popularity"] = hotel_data["review_count"]
review_data.drop(columns = ['hotel_name', 'review_count', 'bubble_score', 'star_rating', 'brand_name', 'parent_brand_name', 'hotel_type', 'city_name'], inplace = True)

for index,row in review_data.iterrows():
    if row["popularity"] > q3:
        review_data["popularity"][index] = 4
    elif row["popularity"] > q2:
        review_data["popularity"][index] = 3
    elif row["popularity"] > q1:
        review_data["popularity"][index] = 2
    else:
        review_data["popularity"][index] = 1

new_merged["hotel_id"] = new_merged["hotel_id"].astype(int)
review_data["hotel_id"] = review_data["hotel_id"].astype(int)

cleaned_data = pd.merge(new_merged, review_data)
#%% Save to file
cleaned_data.to_csv('clean_data.csv', index = False)
