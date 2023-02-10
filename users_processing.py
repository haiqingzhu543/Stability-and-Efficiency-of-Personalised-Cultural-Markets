import numpy as np
import pandas as pd
import pickle
# Processing user groups
users = pd.read_csv("./ml-100k/u.user", sep= '|', header= None, engine= 'python', encoding= 'latin-1')
users.columns = ['Index', 'Age', 'Gender', 'Occupation', 'Zip code']
gender = np.where(np.matrix(users['Gender']) == 'M', 0, 1)[0]
male_index = np.nonzero(np.array(users['Gender']) == 'M')[0]
female_index = np.nonzero(np.array(users['Gender']) == 'F')[0]
#print(male_index)
#print(female_index)
old_index = []
young_index = []


for i in range(len(users['Age'])):
    if int(users['Age'][i]) > 40:
        old_index.append(i)
    else:
        young_index.append(i)


male_old = list(set(old_index).intersection(set(male_index)))
male_young = list(set(young_index).intersection(set(male_index)))
female_old = list(set(old_index).intersection(set(female_index)))
female_young = list(set(young_index).intersection(set(female_index)))

splitted_indices = [np.arange(len(male_index)+len(female_index))]
#splitted_indices = [male_old,male_young,female_old,female_young]

#splitted_indices = []
#for j in range(len(male_index)+len(female_index)):
#    splitted_indices.append([j])

with open("groups","wb") as fp:
    pickle.dump(splitted_indices,fp)

