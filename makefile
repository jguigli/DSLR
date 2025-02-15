datasetURL := https://cdn.intra.42.fr/document/document/12379/datasets.tgz
python := python3

define venvWrapper
	{\
	. bin/activate; \
	$1; \
	}
endef

analysis_dir := src.data_analysis
visualization := src.data_visualization
logReg_dir := src.logistic_regression
dtset_test := datasets/dataset_test.csv
dtset_train := datasets/dataset_train.csv

trainLogReg = $(python) -m $(logReg_dir).logreg_train $1 && \
	$(python) -m $(logReg_dir).logreg_train_stochastic $1 && \
	$(python) -m $(logReg_dir).logreg_train_minibatch $1 \

help:
	@echo "Usage: make [target]"
	@echo "Targets:"
	@echo "  describe:   run the describe script need 1 argument ARG='dataset.csv'"
	@echo "  histogram:  run the histogram script need 1 argument ARG='dataset.csv'"
	@echo "  scatter:    run the scatter plot script need 1 argument ARG='dataset.csv'"
	@echo "  pair:       run the pair plot script need 1 argument ARG='dataset.csv'"
	@echo "	 train:      run the training program need 1 argument ARG='dataset.csv'"
	@echo "	 predict:    run the predict program"
	@echo "  install:    Install the project"
	@echo "  freeze:     Freeze the dependencies"
	@echo "  fclean:     Remove the virtual environment and the datasets"
	@echo "  clean:      Remove the cache files"
	@echo "  re:         Reinstall the project"
	@echo "  phony:      Run the phony targets"

describe:
	@$(call venvWrapper, $(python) -m $(analysis_dir).describe ${ARG})

histogram:
	@$(call venvWrapper, $(python) -m $(visualization).histogram ${ARG})

scatter:
	@$(call venvWrapper, $(python) -m $(visualization).scatter_plot ${ARG})

pair:
	@$(call venvWrapper, $(python) -m $(visualization).pair_plot ${ARG})

train:
	@$(call venvWrapper, $(call trainLogReg, ${ARG}))

predict:
	@$(call venvWrapper, $(python) -m $(logReg_dir).logreg_predict)

install:
	@{ \
		echo "Setting up..."; \
		python3 -m venv .; \
		. bin/activate; \
		if [ -f requirements.txt ]; then \
			pip install -r requirements.txt; \
			echo "Installing dependencies...DONE"; \
		fi; \
		if [ ! -d "datasets" ]; then \
			echo "Downloading datasets..."; \
			wget ${datasetURL}; \
			tar -xvf datasets.tgz; \
			rm -rf datasets.tgz; \
		fi; \
	}

freeze:
	$(call venvWrapper, pip freeze > requirements.txt)

fclean: clean
	rm -rf bin/ include/ lib/ lib64 pyvenv.cfg share/ datasets/

clean:
	rm -rf src/__pycache__ src/*/__pycache__

re: fclean install

phony: install freeze fclean clean re help describe histogram scatter pair train predict