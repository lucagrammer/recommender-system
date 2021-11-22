import os
from typing import Tuple, Callable, Dict, Optional, List

import numpy as np
import pandas as pd
import scipy.sparse as sp

from sklearn.model_selection import train_test_split


def load_data():
    # Fetch the datset
    print("> Importing file...")
    data_file_path = "datasets/"
    data_train_name = data_file_path + "data_train.csv"

    # If directory does not exist, generate error
    if not os.path.exists(data_train_name):
        print("> File not found")

    URM_all_dataframe = pd.read_csv(
        filepath_or_buffer=data_train_name,
        names=["user_id", "item_id", "ratings"],
        sep=",",
        header=0,
        dtype={0: int, 1: int, 2: float},
    )
    print("> Importing file... Completed!")
    return URM_all_dataframe


def load_target():
    # Fetch the datset
    print("> Importing file...")
    data_file_path = "datasets/"
    data_target_users_test_name = data_file_path + "data_target_users_test.csv"

    # If directory does not exist, generate error
    if not os.path.exists(data_target_users_test_name):
        print("> Files not found")

    target_users_test = pd.read_csv(
        filepath_or_buffer=data_target_users_test_name,
        names=["user_id"],
        sep=",",
        header=0,
        dtype={0: int},
    )
    print("> Importing file... Completed!")
    return target_users_test


def split_data(URM_all_dataframe):
    URM_all = sp.coo_matrix(
        (
            URM_all_dataframe["ratings"].values,
            (URM_all_dataframe["user_id"].values, URM_all_dataframe["item_id"].values,),
        )
    )
    train_test_split = 0.80
    n_interactions = URM_all.nnz

    train_mask = np.random.choice(
        [True, False], n_interactions, p=[train_test_split, 1 - train_test_split]
    )
    URM_train = sp.csr_matrix(
        (URM_all.data[train_mask], (URM_all.row[train_mask], URM_all.col[train_mask]),)
    )

    validation_mask = np.logical_not(train_mask)
    URM_validation = sp.csr_matrix(
        (
            URM_all.data[validation_mask],
            (URM_all.row[validation_mask], URM_all.col[validation_mask]),
        )
    )
    return URM_all.tocsr(), URM_train, URM_validation


def prepare_submission(
    ratings: pd.DataFrame, users_to_recommend: np.array, recommender: object,
):
    print("Recommending...")

    recommendation_length = 10
    submission = []
    for user_id in users_to_recommend:
        recommendations = recommender.recommend(
            user_id, cutoff=recommendation_length, remove_seen_flag=True,
        )
        # user_id=mapped_user_id, at=recommendation_length, remove_seen=True,

        submission.append((user_id, [item_id for item_id in recommendations]))
    print("Recommending... Completed!")
    return submission


def write_submission(submissions):
    with open("./submission.csv", "w") as f:
        f.write(f"user_id,item_list")
        for user_id, items in submissions:
            f.write(f"\n{user_id},{' '.join([str(item) for item in items])}")
    print("CSV successfully created.")

