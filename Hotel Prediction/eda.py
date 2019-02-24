#%% Packages

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

#%% Load data

def load_data(path):
    return pd.read_csv(path)

activity_data_original = load_data("activity_data.csv")
hotel_data_original = load_data("hotel_data.csv")

#%% Copy data

activity_data = activity_data_original.copy()
hotel_data = hotel_data_original.copy()

#%%

sns.set(style="whitegrid")
ax = sns.countplot(x = "user_action", hue = "device", data = activity_data)

user_action_device = pd.crosstab(index = activity_data["user_action"], columns = activity_data["device"])
user_action_device.index = ["view","hotel_website_click", "price_click", "booking"]
print(user_action_device)


f, ax = plt.subplots(figsize=(10, 8))
corr = user_action_device.corr()
sns.heatmap(corr, mask = np.zeros_like(corr, dtype = np.bool), cmap = sns.diverging_palette(220, 10, as_cmap=True),square=True, ax=ax)
#%% Fix variables activity data

# device dictionary: 1 desktop computer, 2 mobile browser, 3 mobile app, 0 rows to remove

clean_act_data = {"user_action":{'view':4, 'hotel_website_click':3,'price_click':2, 'booking':1}, "device" : {"windows" : 1, "osx" : 1, "linux" : 1, "iphone_browser" : 2, "iphone_native_app" : 3,"android_browser" : 2, "ipad_browser" : 2, "android_native_app" : 3, "ipad_native_app" : 3, "android_tablet_browser" : 2, "android_tablet_native_app" : 3, "other_phone" : 0, "iphone_hybrid_app" : 3, "ipad_hybrid_app" : 3, "other_tablet" : 0, "android_tablet_hybrid_app" : 3,"android_hybrid_app" : 3, "other" : 0}}

activity_data.replace(clean_act_data, inplace=True)
activity_data = activity_data[activity_data['device'] != 0] 

