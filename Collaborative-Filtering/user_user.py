input_path = ""


'''
for every item i that u has no rating  yet

  for every item j that u has a rating 

	compute a similarity 's' between i and j

	add u's rating for j, weighted by s, to a running average

 return the top items, ranked by weighted average
 '''
from collections import defaultdict
from surprise import SVD
from surprise import Dataset

class UserBasedRecommender(object):
	'''
	User based recommendation Engine
	'''
	def __init__(self, dataset):
		self.dataset =dataset
		self.data = None
		self.trainset = None
		
	def loadDataset(self):
		self.data = Dataset.load_builtin(self.dataset)

	def train(self,algo):
		self.trainset = self.data.build_full_trainset()
		algo.train(self.trainset)

	def predict(self,algo):
		testset = self.trainset.build_anti_testset()
		predictions = algo.test(testset)
		return predictions
	
	def recommendedItems(self,predictions, n=10):
		'''
		Return the top-N recommendation for each user from a set of predictions.
		'''
		top_n = defaultdict(list)
		for uid, iid, true_r, est, _ in predictions:
			top_n[uid].append((iid, est))

		# Then sort the predictions for each user and retrieve the k highest ones.
		for uid, user_ratings in top_n.items():
			user_ratings.sort(key=lambda x: x[1], reverse=True)
			top_n[uid] = user_ratings[:n]

		return top_n

if __name__ == "__main__":

	# load dataset = ml-100k builtin surprise.
	userRecommender = UserBasedRecommender('ml-100k')
	#change it to load custom dataset
	userRecommender.loadDataset()
	algo = SVD()
	userRecommender.train(algo)
	predictions = userRecommender.predict(algo)
	top_n = userRecommender.recommendedItems(predictions,n=2)
	# Print the recommended items for each user
	for uid, user_ratings in top_n.items():
		print(uid, [iid for (iid, _) in user_ratings])
	