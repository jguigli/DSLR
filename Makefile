all: train predict

describe:
	@cd ./data_analysis && python3 describe.py
histogram:
	@cd ./data_visualization && python3 histogram.py
scatter:
	@cd ./data_visualization && python3 scatter_plot.py
pair:
	@cd ./data_visualization && python3 pair_plot.py
train:
	@cd ./logistic_regression && python3 logreg_train.py
predict:
	@cd ./logistic_regression && python3 logreg_predict.py
score : train predict
	@cd ./logistic_regression && python3 accuracy_score.py
clean:
	@cd ./logistic_regression && rm thetas.csv houses.csv