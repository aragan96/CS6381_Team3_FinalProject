data = open("dataset_from_logs.csv",'w')
cores = [2, 4, 8, 12, 16]
data.write("Cores,Iterations,Regularization Parameter,MSE,Seconds\n")

for corenum in cores:
	f = open("logs/pyspark%i.log" % (corenum))
	for line in f.readlines():
		tokens = line.split()
		if tokens[0] == "Iterations:":
			data.write("%i,%s,%s,%s," % (corenum,tokens[1],tokens[4],tokens[6]))
		elif tokens[0] == "Took":
			data.write("%s\n" % (tokens[1]))

data.close()


