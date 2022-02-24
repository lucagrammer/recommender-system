from Recommenders.BaseRecommender import BaseRecommender
import scipy.sparse as sps
from numpy import linalg as LA
import numpy as np


class ScoresHybrid3Recommender(BaseRecommender):
    """ ScoresHybrid3Recommender

    """

    RECOMMENDER_NAME = "ScoresHybrid3Recommender"

    def __init__(self, URM_train, recommender_1, recommender_2, recommender_3):
        super(ScoresHybrid3Recommender, self).__init__(URM_train)

        self.URM_train = sps.csr_matrix(URM_train)
        self.recommender_1 = recommender_1
        self.recommender_2 = recommender_2
        self.recommender_3 = recommender_3

    def fit(self, a=0.33, b=0.33, c=0.33, a1=0.7535):
        self.a = a
        self.b = b
        self.c = c
        self.a1 = a1

    def _compute_item_score(self, user_id_array, items_to_compute):

        item_weights = np.empty([len(user_id_array), 18059])
        for i in range(len(user_id_array)):

            interactions = len(self.URM_train[user_id_array[i], :].indices)

            if interactions < 8:
                w1 = self.recommender_1._compute_item_score(
                    user_id_array[i], items_to_compute
                )
                w1 /= LA.norm(w1, 2)
                w = w1
                item_weights[i, :] = w
            elif interactions > 7 and interactions < 18:
                w1 = self.recommender_1._compute_item_score(
                    user_id_array[i], items_to_compute
                )
                w2 = self.recommender_2._compute_item_score(
                    user_id_array[i], items_to_compute
                )
                w1 /= LA.norm(w1, 2)
                w2 /= LA.norm(w2, 2)
                w = w1 * self.a1 + w2 * (1 - self.a1)
                item_weights[i, :] = w
            else:
                w1 = self.recommender_1._compute_item_score(
                    user_id_array[i], items_to_compute
                )
                w2 = self.recommender_2._compute_item_score(
                    user_id_array[i], items_to_compute
                )
                w3 = self.recommender_3._compute_item_score(
                    user_id_array[i], items_to_compute
                )
                w1 /= LA.norm(w1, 2)
                w2 /= LA.norm(w2, 2)
                w3 /= LA.norm(w3, 2)
                w = w1 * self.a + w2 * self.b + w3 * self.c
                item_weights[i, :] = w

        return item_weights
