"""
@author: Nicholas Gao
@co-author: Ryan Jaipersaud
Frequentist Machine Learning
Professor Keene
Final MiniProject
"""

from typing import List, Tuple

from surprise import SVD
from surprise import Dataset
from surprise import accuracy
from surprise import Reader, get_dataset_dir
from collections import defaultdict
import os
import io


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
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_k[uid] = user_ratings[:k]
    return top_k


def read_item_names():
    """Read the u.item file from MovieLens 100-k dataset and return two
    mappings to convert raw ids into movie names and movie names into raw ids.
    """

    file_name = get_dataset_dir() + '/ml-100k/ml-100k/u.item'
    rid_to_name = {}
    name_to_rid = {}
    with io.open(file_name, 'r', encoding='ISO-8859-1') as f:
        for line in f:
            line = line.split('|')
            rid_to_name[line[0]] = line[1]
            name_to_rid[line[1]] = line[0]

    return rid_to_name, name_to_rid


def get_user_recs(uid_for_new_user, dev=True):
    """
            Get recommendations from the user using this recommendation system
            :param uid_for_new_user: user id for new user
            :param dev: TRUE -> Use preset responses
            :return: list of tuples [(uid, mid, rating, timestamp)...]
        """
    movies = ["Casablanca", "Toy Story", "Pulp Fiction", "Star Wars", "Forrest Gump"]
    m_id = ['483', '1', '56', '50', '69']
    dev_answers = [2.0, 4.0, 3.0, 5.0, 4.0]
    input_list: List[Tuple[str, str, float, str]] = []
    time = '891295330'

    if dev:
        for i in range(0, 5):
            my_tuple = (uid_for_new_user, m_id[i], dev_answers[i], time)
            input_list.append(my_tuple)
    else:
        print("Please rate the following movies with a rating from 1 (not interested) to 5 (really like): ")
        # Create list of tuples to append to test set
        for i in range(0, 5):
            temp_input = input("Rating for " + movies[i] + ": ")
            while int(temp_input) not in [1, 2, 3, 4, 5]:
                temp_input = input("Rating for " + movies[i] + ": ")

            my_tuple = (uid_for_new_user, m_id[i], float(temp_input), time)
            input_list.append(my_tuple)

    return input_list

if __name__ == "__main__":
    # Get Data
    # file_path = os.path.expanduser('C:/cygwin64/home/jaipe/Machine Learning/ml-100k/u.data')
    file_path = os.path.expanduser('/Users/NicholasGao/Downloads/ml-100k/u.data')
    reader = Reader(line_format='user item rating timestamp', sep='\t')
    data = Dataset.load_from_file(file_path, reader=reader)


    # Get the mappings: raw id <-> movie name
    rid_to_name, name_to_rid = read_item_names()

    # Ask for user ratings
    new_uid = '1500'
    my_input_list = get_user_recs(uid_for_new_user=new_uid, dev=False)

    # Add to new user data to train on
    for tup in my_input_list:
        data.raw_ratings.append(tup)

    # First train an SVD algorithm on the movielens dataset.
    trainset = data.build_full_trainset()
    algo = SVD()  # SVD algorithm.
    algo.fit(trainset)

    # Should be old number of users + 1 for new user of this recommendaiton app
    print("num users: ", trainset.n_users)
    # print("num ratings: ", trainset.n_ratings)
    # print("num items: ", trainset.n_items)

    # ============================================================================

    # Then, Predict ratings for all pairs (u, i) that are NOT in the training set.
    testset = trainset.build_anti_testset()
    predictions = algo.test(testset)

    print("Accuracy below is not applicable since this is the production version, "
          "where testset is the actual things we want to predict (i.e. there are no true user ratings for this dataset).")
    accuracy.rmse(predictions)  # compute RMSE
    top_k = get_top_k(predictions, 5)  # get the 5 best predictions for each user

    # Print the recommended items
    for uid, user_ratings in top_k.items():
        if uid == new_uid:
            print("Your predicted movies: ")
            print(uid, [iid for (iid, _) in user_ratings])
            print([rid_to_name[iid] for (iid, _) in user_ratings])