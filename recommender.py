from pyspark import SparkContext
from pyspark.mllib.recommendation import ALS, Rating
import time

# Code draws heavily from example code at https://github.com/apache/spark/blob/master/examples/src/main/python/mllib/recommendation_example.py

def build_collaborative_filtering_model(file_location, numIterations = 10, regularization_parameter = 0.1):
	# file_location: Location of a file containing dataset formatted as 
	#				 User ID, User Item, rating

	# Format data
	context = SparkContext(appName="collaborative_filtering_model")
	data = context.textFile(file_location)
	instances = data.map(lambda instance: instance.split(","))
	ratings = instances.map(lambda arr: Rating(int(arr[0]), int(arr[1]), float(arr[2])))

	# Build model
	model = ALS.train(ratings, 10, numIterations, regularization_parameter)
	test_set = ratings.map(lambda p: (p[0], p[1]))
	predictions = model.predictAll(test_set).map(lambda r: ((r[0], r[1]), r[2]))
	ratesAndPreds = ratings.map(lambda r: ((r[0], r[1]), r[2])).join(predictions)
	MSE = ratesAndPreds.map(lambda r: (r[1][0] - r[1][1])**2).mean()
	return MSE

def run_collaborative_filtering_with_variable_hyperparameters(file_location):
	iterations_to_test = [1, 5, 10, 20, 50]
	regularization_parameters_to_test = [0.01, 0.1, 1, 10]
	for iterations in iterations_to_test:
		for regularization_parameter in regularization_parameters_to_test:
			start = time.time()
			MSE = build_collaborative_filtering_model(file_location, iterations, regularization_parameter)
			end = time.time()
			dif = end-start
			print("Iterations: " + str(iterations) + " Regularization parameter: " + str(regularization_parameter) + " MSE: " + str(MSE))
			print("Took " + str(dif) + " seconds")

if __name__ == "__main__":
	data_file_location = "example.csv"
	run_collaborative_filtering_with_variable_hyperparameters(data_file_location)
