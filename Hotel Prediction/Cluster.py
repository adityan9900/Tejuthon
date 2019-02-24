import pandas as pd
import matplotlib.pyplot as plt
import math
import numpy as np
import sklearn as scikit_learn
import sklearn.cluster as cluster

data = pd.read_csv("./clean_data.csv")
data.drop(columns = ['date', 'user_id', 'user_country', 'device', 
                     'hotel_id', 'user_action', 'city_name', 'hotel_type', 
                     'parent_brand_name', 'brand_name', 'review_count',  'user_country_enc', 'device_enc', 
                     'user_action_enc', 'brand_name_enc', 'hotel_name'], inplace = True)
data = data.fillna(0)
kmeans = cluster.KMeans()
kmeans.fit(data.sample(10000))
print(kmeans.cluster_centers_)

