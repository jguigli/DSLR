all: describe histogram scatter plot train predict

describe:
	@python3 ./data_analysis/training_model.py
histogram:
	@python3 ./data_visualization/predict.py
scatter:
	@python3 ./data_visualization/graph.py
plot:
	@python3 ./data_visualization/algorithm_accuracy.py
train:
	@python3 ./logistic_regression/algorithm_accuracy.py
predict:
	@python3 ./logistic_regression/algorithm_accuracy.py
clean:
	@rm ../data/thetas.csv