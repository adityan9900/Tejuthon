import pandas as pd
import numpy as np

dictionary_df = pd.read_excel('/home/adi/Downloads/dictionary.xlsx')        
user_profile_df = pd.read_csv('/home/adi/Downloads/user_profile.csv', dtype = 'unicode')
#first_session_df = pd.read_csv(first_session_file)
user_engagement_df = pd.read_csv('/home/adi/Downloads/user_engagement.csv', dtype = 'unicode')

user_ids = user_profile_df['user_id']
user_ids_eng = user_engagement_df['user_id']


user_id_indexes = {} #a dictionary with key of user_id and value of list of indexes where user_id is found in user_engagement_id
z = 0
print("total users: " , len(user_ids))
user_ids = user_ids[18000:20000]

for user_id in user_ids:
    print("current ID num: ", z)
    user_indices = [i for i, x in enumerate(user_ids_eng) if x == user_id]
    click_count_list = list()
    click_count_credit_card_list = list()
    click_count_personal_loan_list = list()
    click_count_mortgage_list = list()
    click_count_credit_repair_list = list()
    click_count_banking_list = list()
    click_count_auto_products_list = list()
    
    click_apply_count_list = list()
    click_apply_count_credit_card_list = list()
    click_apply_count_personal_loan_list = list()
    click_apply_count_mortgage_list = list()
    click_apply_count_credit_repair_list = list()
    click_apply_count_banking_list = list()
    click_apply_count_auto_products_list = list()
    
    
    for i in user_indices:
       # print("adding another value")
        click_count_list.append(float(user_engagement_df['click_count'][i]))
        click_count_credit_card_list.append(float(user_engagement_df['click_count_credit_card'][i]))
        click_count_personal_loan_list.append(float(user_engagement_df['click_count_personal_loan'][i]))
        click_count_mortgage_list.append(float(user_engagement_df['click_count_mortgage'][i]))
        click_count_credit_repair_list.append(float(user_engagement_df['click_count_credit_repair'][i]))
        click_count_banking_list.append(float(user_engagement_df['click_count_banking'][i]))
        click_count_auto_products_list.append(float(user_engagement_df['click_count_auto_products'][i]))
        
        click_apply_count_list.append(float(user_engagement_df['click_apply_count'][i]))
        click_apply_count_credit_card_list.append(float(user_engagement_df['click_apply_count_credit_card'][i]))
        click_apply_count_personal_loan_list.append(float(user_engagement_df['click_apply_count_personal_loan'][i]))
        click_apply_count_mortgage_list.append(float(user_engagement_df['click_apply_count_mortgage'][i]))
        click_apply_count_credit_repair_list.append(float(user_engagement_df['click_apply_count_credit_repair'][i]))
        click_apply_count_banking_list.append(float(user_engagement_df['click_apply_count_banking'][i]))
        click_apply_count_auto_products_list.append(float(user_engagement_df['click_apply_count_auto_products'][i]))
        
        
        
        
    click_count_array = np.array(click_count_list)
    click_count_credit_card_array = np.array(click_count_credit_card_list)
    click_count_personal_loan_array = np.array(click_count_personal_loan_list)
    click_count_mortgage_array = np.array(click_count_mortgage_list)
    click_count_credit_repair_array = np.array(click_count_credit_repair_list)
    click_count_banking_array = np.array( click_count_banking_list)
    click_count_auto_products_array = np.array(click_count_auto_products_list)
    
    click_apply_count_array = np.array(click_apply_count_list)
    click_apply_count_credit_card_array = np.array(click_apply_count_credit_card_list)
    click_apply_count_personal_loan_array = np.array(click_apply_count_personal_loan_list)
    click_apply_count_mortgage_array = np.array(click_apply_count_mortgage_list)
    click_apply_count_credit_repair_array = np.array(click_apply_count_credit_repair_list)
    click_apply_count_banking_array = np.array( click_apply_count_banking_list)
    click_apply_count_auto_products_array = np.array(click_apply_count_auto_products_list)
    
    if(len(click_count_array) == 0):
        click_count_average = 0
    else:
        click_count_average = np.mean(click_count_array)
        
    if(len(click_count_credit_card_array) == 0):
        click_count_credit_card_average = 0
    else:
        click_count_credit_card_average = np.mean(click_count_credit_card_array)
        
    if(len(click_count_personal_loan_array) == 0):
        click_count_personal_loan_average = 0
    else:
        click_count_personal_loan_average = np.mean(click_count_personal_loan_array)
    
    if(len(click_count_mortgage_array) == 0):
        click_count_credit_card_average = 0
    else:
       click_count_mortgage_average = np.mean(click_count_mortgage_array)
    
    if(len(click_count_credit_repair_array) == 0):
        click_count_credit_repair_average = 0
    else:
       click_count_credit_repair_average = np.mean(click_count_credit_repair_array)
    
    
    if(len(click_count_banking_array) == 0):
        click_count_banking_average = 0
    else:
       click_count_banking_average = np.mean(click_count_banking_array)
    
    if(len(click_count_auto_products_array) == 0):
        click_count_auto_products_average = 0
    else:
       click_count_auto_products_average = np.mean(click_count_auto_products_array)
        
        
     
    if(len(click_apply_count_array) == 0):
        click_apply_count_average = 0
    else:
        click_apply_count_average = np.mean(click_apply_count_array)
        
    if(len(click_apply_count_credit_card_array) == 0):
        click_apply_count_credit_card_average = 0
    else:
        click_apply_count_credit_card_average = np.mean(click_apply_count_credit_card_array)
        
    if(len(click_apply_count_personal_loan_array) == 0):
        click_apply_count_personal_loan_average = 0
    else:
        click_apply_count_personal_loan_average = np.mean(click_apply_count_personal_loan_array)
    
    if(len(click_apply_count_mortgage_array) == 0):
        click_apply_count_credit_card_average = 0
    else:
       click_apply_count_mortgage_average = np.mean(click_apply_count_mortgage_array)
    
    if(len(click_apply_count_credit_repair_array) == 0):
        click_apply_count_credit_repair_average = 0
    else:
       click_apply_count_credit_repair_average = np.mean(click_apply_count_credit_repair_array)
    
    
    if(len(click_apply_count_banking_array) == 0):
        click_apply_count_banking_average = 0
    else:
       click_apply_count_banking_average = np.mean(click_apply_count_banking_array)
    
    if(len(click_apply_count_auto_products_array) == 0):
        click_apply_count_auto_products_average = 0
    else:
       click_apply_count_auto_products_average = np.mean(click_apply_count_auto_products_array)
     
    
    user_id_indexes[user_id] = list()
    user_id_indexes[user_id].append(float(click_count_average))
    user_id_indexes[user_id].append(float(click_count_credit_card_average))
    user_id_indexes[user_id].append(float(click_count_personal_loan_average))
    user_id_indexes[user_id].append(float(click_count_mortgage_average))
    user_id_indexes[user_id].append(float(click_count_credit_repair_average))
    user_id_indexes[user_id].append(float(click_count_banking_average))
    user_id_indexes[user_id].append(float(click_count_auto_products_average))
    
    
    user_id_indexes[user_id].append(float(click_apply_count_average))
    user_id_indexes[user_id].append(float(click_apply_count_credit_card_average))
    user_id_indexes[user_id].append(float(click_apply_count_personal_loan_average))
    user_id_indexes[user_id].append(float(click_apply_count_mortgage_average))
    user_id_indexes[user_id].append(float(click_apply_count_credit_repair_average))
    user_id_indexes[user_id].append(float(click_apply_count_banking_average))
    user_id_indexes[user_id].append(float(click_apply_count_auto_products_average))
    z += 1



    f = open('/home/adi/labels_5.csv','w')

    for el in user_id_indexes.keys():
    	f.write(str(user_id_indexes[el])[1:-1] + "\n") #Give your csv text here.
	
    f.close()

