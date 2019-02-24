import pandas as pd
pd.options.mode.chained_assignment = None
hotel_data_original = pd.read_csv("hotel_data.csv")
review_data = hotel_data_original

q1 = review_data.review_count.quantile([0.25,0.5,0.75])[0.25]
q2 = review_data.review_count.quantile([0.25,0.5,0.75])[0.5]
q3 = review_data.review_count.quantile([0.25,0.5,0.75])[0.75]

review_data.loc[:, "popularity"] = review_data["review_count"]


for index,row in review_data.iterrows():
    if row["popularity"] > q3:
        review_data["popularity"][index] = 4
    elif row["popularity"] > q2:
        review_data["popularity"][index] = 3
    elif row["popularity"] > q1:
        review_data["popularity"][index] = 2
    else:
        review_data["popularity"][index] = 1

def find_hotel(star, bubble, h_type, city_name, p_brand_name):
    hotel_data = pd.merge(hotel_data_original, review_data)

    hotel_data = hotel_data[hotel_data['city_name'] == city_name]
    hotel_data = [hotel_data['hotel_type'] == h_type]
    hotel_data = hotel_data[hotel_data['parent_brand_name'] == p_brand_name]
    hotel_data = hotel_data[hotel_data['star_rating'] == star]
    hotel_data = hotel_data[hotel_data['bubble_score'] == bubble]
    
    max_pop = hotel_data['popularity'].max
    
    hotel_data = hotel_data[hotel_data['popularity'] == max_pop]
    
    return hotel_data.iloc[:3,] 