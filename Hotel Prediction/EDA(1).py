import pandas as pd
import matplotlib.pyplot as plt
import math
import numpy as np
import operator 

activity_data = pd.read_csv("./activity_data.csv")
original_activity = pd.read_csv("./activity_data.csv")
hotel_data = pd.read_csv("./hotel_data.csv")
temp_hotel = pd.read_csv("./hotel_data.csv")

clean = {"device" : {"windows" : 1, "osx" : 1, "linux" : 1, "iphone_browser" : 2, "iphone_native_app" : 3, "android_browser" : 2, "ipad_browser" : 2, "android_native_app" : 3, "ipad_native_app" : 3, "android_tablet_browser" : 2, "android_tablet_native_app" : 3, "other_phone" : 0, "iphone_hybrid_app" : 3, "ipad_hybrid_app" : 3, "other_tablet" : 0, "android_tablet_hybrid_app" : 3, "android_hybrid_app" : 3, "other" : 0}, 
         "user_action": {"booking" : 1, "price_click" : 2, "hotel_website_click" : 3, "view" : 4}}

activity_data.replace(clean, inplace = True)
q1 = hotel_data.review_count.quantile([0.25,0.5,0.75])[0.25]
q2 = hotel_data.review_count.quantile([0.25,0.5,0.75])[0.5]
q3 = hotel_data.review_count.quantile([0.25,0.5,0.75])[0.75]

hotel_data.loc[:, "view_quant"] = hotel_data["review_count"]
hotel_data.drop(columns = ['hotel_name', 'review_count', 'bubble_score', 'star_rating', 'brand_name', 'parent_brand_name', 'hotel_type', 'city_name'], inplace = True)

for index,row in hotel_data.iterrows():
    if row["view_quant"] > q3:
        hotel_data["view_quant"][index] = 4
    elif row["view_quant"] > q2:
        hotel_data["view_quant"][index] = 3
    elif row["view_quant"] > q1:
        hotel_data["view_quant"][index] = 2
    else:
        hotel_data["view_quant"][index] = 1
hotel_data.to_csv('quartiles.csv', index = False)
#
#activity_data = activity_data[activity_data.device != 0]
#import seaborn as sns
#sns.distplot(hotel_data['review_count'],kde = True)
#plt.show()
#corr = hotel_data.corr()
#
## Generate a mask for the upper triangle
#mask = np.zeros_like(corr, dtype=np.bool)
#mask[np.triu_indices_from(mask)] = True
#
## Set up the matplotlib figure
#f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
#cmap = sns.diverging_palette(275, 0, s=75, l=40, as_cmap = True)

# Draw the heatmap with the mask and correct aspect ratio
#sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3,
            #square=True, linewidths=.5, cbar_kws={"shrink": .5})
# plt.show()
#hotel_data.loc[:, "ratings"] = hotel_data["bubble_score"] *  hotel_data["review_count"]
#brandrev = {}
#brandn = {}
#parn = {}
#parrev = {}
#typerev = {}
#typen = {}
#for index,row in hotel_data.iterrows():
#    if row["brand_name"] not in brandrev and pd.Series([row["brand_name"]]).notnull()[0]:
#        brandrev[row["brand_name"]] = row["review_count"]
#        brandn[row["brand_name"]] = 1
#    elif pd.Series([row["brand_name"]]).notnull()[0]:
#        brandrev[row["brand_name"]] = brandrev[row["brand_name"]] + row["review_count"]
#        brandn[row["brand_name"]] += 1
#    if row["parent_brand_name"] not in parrev and pd.Series([row["parent_brand_name"]]).notnull()[0]:
#        parrev[row["parent_brand_name"]] = row["review_count"]
#        parn[row["parent_brand_name"]] = 1
#    elif pd.Series([row["parent_brand_name"]]).notnull()[0]:
#        parrev[row["parent_brand_name"]] = parrev[row["parent_brand_name"]] + row["review_count"]
#        parn[row["parent_brand_name"]] += 1
#    if row["hotel_type"] not in typerev:
#        typerev[row["hotel_type"]] = row["review_count"]
#        typen[row["hotel_type"]] = 1
#    else:
#        typerev[row["hotel_type"]] = typerev[row["hotel_type"]] + row["review_count"]
#        typen[row["hotel_type"]] += 1
#brandrev.pop("None")
#brandrev = dict(sorted(brandrev.items(), key=operator.itemgetter(1), reverse=True)[:5])
#parrev = dict(sorted(parrev.items(), key=operator.itemgetter(1), reverse=True)[:5])
#typerev = dict(sorted(typerev.items(), key=operator.itemgetter(1), reverse=True)[:5])
#
#bstar = {}
#bbub = {}
#pstar = {}
#pbub = {}
#tstar = {}
#tbub = {}
#for index,row in hotel_data.iterrows():
#    if row["brand_name"] in brandrev:
#        if not math.isnan(row["star_rating"]):
#            if row["brand_name"] not in bstar:
#                bstar[row["brand_name"]] = row["star_rating"]
#            else:
#                bstar[row["brand_name"]] += row["star_rating"]
#        if not math.isnan(row["bubble_score"]):
#            if row["brand_name"] not in bbub:
#                bbub[row["brand_name"]] = row["bubble_score"]
#            else:
#                bbub[row["brand_name"]] += row["bubble_score"]
#    if row["parent_brand_name"] in parrev:
#        if not math.isnan(row["star_rating"]):
#            if row["parent_brand_name"] not in pstar:
#                pstar[row["parent_brand_name"]] = row["star_rating"]
#            else:
#                pstar[row["parent_brand_name"]] += row["star_rating"]
#        if not math.isnan(row["bubble_score"]):
#            if row["parent_brand_name"] not in pbub:
#                pbub[row["parent_brand_name"]] = row["bubble_score"]
#            else:
#                pbub[row["parent_brand_name"]] += row["bubble_score"]
#    if row["hotel_type"] in typerev:
#        if not math.isnan(row["star_rating"]):
#            if row["hotel_type"] not in tstar:
#                tstar[row["hotel_type"]] = row["star_rating"]
#            else:
#                tstar[row["hotel_type"]] += row["star_rating"]
#        if not math.isnan(row["bubble_score"]):
#            if row["hotel_type"] not in tbub:
#                tbub[row["hotel_type"]] = row["bubble_score"]
#            else:
#                tbub[row["hotel_type"]] += row["bubble_score"]
#for i in pstar:
#    pstar[i] = pstar[i] / parn[i]
#for i in pbub:
#    pbub[i] = pbub[i] / parn[i]
#for i in tstar:
#    tstar[i] = tstar[i] / typen[i]
#for i in tbub:
#    tbub[i] = tbub[i] / typen[i]
#for i in bstar:
#    bstar[i] = bstar[i] / brandn[i]
#for i in bbub:
#    bbub[i] = bbub[i] / brandn[i]
#plt.bar(range(len(tbub)), list(tbub.values()), align='center')
#plt.xticks(range(len(tbub)), list(tbub.keys()))
#ax.set_xlabel('Hotel Type')
#ax.set_ylabel('Average Bubble Score')
#plt.title("Most Popular Type of Hotel and Average Bubble Score")
#plt.show()


#totrev = {}
#hotrev = {}
#
#for i in range(2, 11):
#    sum = 0
#    tot = 0
#    s = 0
#    utot = 0
#    hotamt = 0
#    for index,row in hotel_data.iterrows():
#        if row["star_rating"] == i * 0.5 and not math.isnan(row["ratings"]):
#            sum += row["ratings"]
#            tot += row["review_count"]
#            hotamt += 1
#    avg = sum/tot
#    for index,row in hotel_data.iterrows():
#        if row["star_rating"] == i * 0.5 and not math.isnan(row["ratings"]):
#            for j in range(0, row["review_count"]):
#                s = s + (row["bubble_score"] - avg)**2
#    s = (s/(tot-1)) ** (1/2) / (tot ** (1/2))
#    hotrev[i*0.5] = hotamt
#    totrev[i*0.5] = tot
#temp = hotel_data["star_rating"]

