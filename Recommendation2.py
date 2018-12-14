# -*- coding: utf-8 -*-
"""

Created on Wed Dec 12 13:28:14 2018
http://surpriselib.com/
https://anaconda.org/conda-forge/scikit-surprise
https://anaconda.org/anaconda/cython
https://github.com/NicolasHug/Surprise/issues/22
@author: jaipe
"""

from surprise import SVD
from surprise import Dataset
from surprise import accuracy
from surprise import Reader
from surprise.model_selection import train_test_split
from collections import defaultdict

def get_top_k(predictions, k):
    '''Return a top_k dicts where keys are user ids and values are lists of
    tuples [(item id, rating estimation) ...].

    Takes in a list of predictions as returned by the test method.
    '''

    # First map the predictions to each user.
    top_k = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_k[uid].append((iid, est))

    # Then sort the predictions for each user and retrieve the k highest ones.
    for uid, user_ratings in top_k.items():
        user_ratings.sort(key=lambda x:x[1], reverse=True)
        top_k[uid] = user_ratings[:k]
    return top_k


file_path = os.path.expanduser('C:/cygwin64/home/jaipe/Machine Learning/ml-100k/u.data')
reader = Reader(line_format='user item rating timestamp', sep='\t')
data = Dataset.load_from_file(file_path, reader=reader)

trainset, testset = train_test_split(data, test_size=0.25)# splits the training set into 75 % train and 25 % test
print(trainset.n_users)
print(trainset.n_ratings)
print(trainset.n_items)

algo = SVD()#SVD algorithm.
        
# Train the algorithm on the trainset, and predict ratings for the testset
algo.fit(trainset)
predictions = algo.test(testset)
accuracy.rmse(predictions)# compute RMSE
top_k = get_top_k(predictions, 5) # get the 5 best predictions for each user

# Print the recommended items
for uid, user_ratings in top_k.items():
    if uid == '160':
        print(uid, [iid for (iid, _) in user_ratings])
        
        
        



