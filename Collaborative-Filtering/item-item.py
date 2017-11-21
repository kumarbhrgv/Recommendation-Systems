from surprise import Dataset, evaluate
from surprise import KNNBasic
from collections import defaultdict
import os, io


class item_item_Recommender():

    def __init__(self,options):
        self.trainingSet = None
        self.knn = KNNBasic(sim_options=options)

    def loadDataset(self):
        data = Dataset.load_builtin("ml-100k")
        self.trainingSet = data.build_full_trainset()

    def train(self):
        self.knn.train(self.trainingSet)

    def test(self):
        testSet = self.trainingSet.build_anti_testset()
        self.predictions = self.knn.test(testSet)
        return  self.predictions

    def get_recommendations(self, topN=3):
        top_recs = defaultdict(list)
        for uid, iid, true_r, est, _ in self.predictions:
            top_recs[uid].append((iid, est))
        for uid, user_ratings in top_recs.items():
            user_ratings.sort(key=lambda x: x[1], reverse=True)
            top_recs[uid] = user_ratings[:topN]
        return top_recs


def read_item_names():
    """
    Read the u.item file from MovieLens 100-k dataset and returns a
    mapping to convert raw ids into movie names.
    """
    file_name = (os.path.expanduser('~') +'/.surprise_data/ml-100k/ml-100k/u.item')
    rid_to_name = {}
    with io.open(file_name, 'r', encoding='ISO-8859-1') as f:
        for line in f:
            line = line.split('|')
            rid_to_name[line[0]] = line[1]
    return rid_to_name


if __name__ == "__main__":
    sim_options = {
        'name': 'cosine',
        'user_based': False
    }
    itemRecommender = item_item_Recommender(sim_options)
    itemRecommender.loadDataset()
    print("training the model")
    itemRecommender.train()
    print("testing the model")
    predictions = itemRecommender.test()
    top3_recommendations = itemRecommender.get_recommendations(3)
    rid_to_name = read_item_names()
    for uid, user_ratings in top3_recommendations.items():
        print(uid, [rid_to_name[iid] for (iid, _) in user_ratings])
