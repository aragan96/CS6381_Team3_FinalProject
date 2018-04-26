from pyspark import SparkContext

from pyspark.mllib.recommendation import ALS, Rating

# Code draws heavily from example code at https://github.com/apache/spark/blob/master/examples/src/main/python/mllib/recommendation_example.py

def build_collaborative_filtering_model(file_location):
	# file_location: Location of a file containing dataset formatted as 
	#				 User ID, User Item, rating

	# Format data
	context = SparkContext(appName="collaborative_filtering_model")
	data = context.textFile(file_location)
	instances = data.map(lambda instance: instance.split(","))
	ratings = instances.map(lambda arr: Rating(int(l[0], int(l[1]), float(l[2]))))

	# Build model
	rank = 10
	numIterations = 10
	model = ALS.train(ratings, rank, numIterations)
	test_set = ratings.map(lambda p: (p[0], p[1]))
	predictions = model.predictAll(test_set).map(lambda r: ((r[0], r[1]), r[2]))
	ratesAndPreds = ratings.map(lambda r: ((r[0], r[1]), r[2])).join(predictions)
	MSE = ratesAndPreds.map(lambda r: (r[1][0] - r[1][1])**2).mean()
	print("Mean Squared Error: " + str(MSE))

if __name__ == "__main__":
	data_file_location = "example.csv"
	build_collaborative_filtering_model(data_file_location)