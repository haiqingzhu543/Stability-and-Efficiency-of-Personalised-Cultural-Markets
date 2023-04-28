from surprise import accuracy, Dataset, SVD, BaselineOnly
from surprise.model_selection import train_test_split
import numpy as np
import pickle
import pandas as pd
from surprise.model_selection import cross_validate
from sklearn.cluster import KMeans

# Load the movielens-100k dataset (download it if needed),
data = Dataset.load_builtin("ml-100k")

# sample random trainset and testset
trainset = data.build_full_trainset()

train_set_test = trainset.build_testset()
untrain_set = trainset.build_anti_testset()


# We'll use the famous SVD algorithm
algo = SVD(n_factors =100)

cross_validate(algo,data,measures=["RMSE","MAE"],cv=5,verbose=True)
algo.fit(trainset)
user_size = 943
item_size = 1682

complete_matrix = np.zeros((user_size,item_size))
untrained_mask = np.zeros((user_size,item_size))

predictions_training = algo.test(train_set_test)
predictions_testing = algo.test(untrain_set)


for uid, iid, true_r, est, _ in predictions_training:
    complete_matrix[int(uid)-1,int(iid)-1] = true_r

for uid, iid, true_r, est, _ in predictions_testing:
    complete_matrix[int(uid)-1,int(iid)-1] = est
    untrained_mask[int(uid)-1,int(iid)-1] = 1


np.save('full_rating_matrix.npy',complete_matrix)
np.save('untrained_mask.npy',untrained_mask)

n_groups = 200
# Group processing_argmax method
"""
user_attributes = np.asarray(algo.pu)
groups = []
for j in range(n_groups):
    groups.append([])
for j in range(user_attributes.shape[0]):
    groups[np.argmax(user_attributes[j,:n_groups])].append(j)
for i in range(len(groups)):
    try:
        groups.remove([])
    except:
        break
print(len(groups))
print(groups)
with open("groups","wb") as fp:
    pickle.dump(groups,fp)
"""



# Group processing Kmeans method
user_attributes = np.asarray(algo.pu)
kmeans = KMeans(n_clusters = n_groups )
kmeans.fit(complete_matrix)
groups = []
for j in range(n_groups):
    groups.append([])
for j in range(user_attributes.shape[0]):
    groups[kmeans.labels_[j]].append(j)
with open("groups","wb") as fp:
    pickle.dump(groups,fp)
