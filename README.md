# Data Science Logistic Regression

This project provides a comprehensive toolkit for data analysis and visualization, with a focus on logistic regression models. It includes modules for data loading, statistical analysis, and various visualization techniques.

## Features

- Load and save CSV files with ease.
- Perform statistical analysis on datasets.
- Train logistic regression models using different gradient descent algorithms.
- Visualize data with scatter plots, pair plots, and histograms.

## Setup

### Prerequisites

- Python 3.7 or higher
- [Pandas](https://pandas.pydata.org/)
- [NumPy](https://numpy.org/)
- [Matplotlib](https://matplotlib.org/)
- [Seaborn](https://seaborn.pydata.org/)

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/jguigli/DSLR.git
   cd DSLR
   ```

2. Install the required packages in the virtual environment:

   ```bash
   make install
   ```

## Usage

### Data Analysis

- **Describe Dataset**: Calculate various statistical metrics for numerical data.

  ```bash
  make describe ARG=<path_to_dataset>
  ```

### Logistic Regression

- **Train Model**: Train a logistic regression model using batch, stochastic, or mini-batch gradient descent.

  ```bash
  make train ARG=<path_to_dataset>
  ```

- **Predict**: Use the trained model to predict outcomes.

  ```bash
  make predict
  ```

### Data Visualization

- **Scatter Plot**: Create scatter plots for numerical data.

  ```bash
  make scatter ARG=<path_to_dataset>
  ```

- **Pair Plot**: Generate pair plots to visualize relationships between variables.

  ```bash
  make pair ARG=<path_to_dataset>
  ```

- **Histogram**: Create histograms to visualize the distribution of numerical data.

  ```bash
  make histogram ARG=<path_to_dataset>
  ```

## Features for logistic regression

The dataset contains the following features:
- Index
- Hogwarts House 
- First Name
- Last Name
- Birthday
- Best Hand
- Arithmancy
- Astronomy
- Herbology
- Defense Against the Dark Arts
- Divination
- Muggle Studies
- Ancient Runes
- History of Magic
- Transfiguration
- Potions
- Care of Magical Creatures
- Charms
- Flying

We achieved an accuracy score of **98.75%** by dropping these features:
- Arithmancy
- Defense Against the Dark Arts
- Transfiguration
- Potions
- Care of Magical Creatures

## Algorithm

### Batch Gradient Descent
Parameters are updated after computing the gradient of the error with respect to the entire training set.
- Results in smooth updates in the model parameters

### Stochastic Gradient Descent
Parameters are updated after computing the gradient of the error with respect to a single training example.
- Results in very noisy updates in the parameters

### Mini-Batch Gradient Descent
Parameters are updated after computing the gradient of the error with respect to a subset of the training set.
- Updates can be made less noisy depending on batch size
- Larger batch size results in less noisy updates

## Bonus fields for describe.py

### Mode
The mode is the value that appears most frequently in the dataset.

### Skewness
Skewness measures the asymmetry of the data distribution relative to the mean. It can be helpful in identifying asymmetric distributions:
- **Positive**: More high values than low values relative to the mean
- **Negative**: More low values than high values relative to the mean

### Kurtosis
Kurtosis measures the shape of the data distribution relative to the normal distribution. It can reveal heavy or light tails compared to the normal distribution:
- **Positive** (>3): The distribution has thicker tails (more extreme values) and a sharper peak than the normal distribution
- **Negative** (<3): The distribution has thinner tails (fewer extreme values) and a less sharp peak than the normal distribution
- **Neutral** (=3): Corresponds to a normal distribution

## Useful Links

[SUBJECT](https://cdn.intra.42.fr/pdf/pdf/66152/en.subject.pdf)  
[PANDAS GETTING STARTED](https://pandas.pydata.org/docs/getting_started/index.html#getting-started)  
[Multi-classifier Logistic Regression](https://www.cs.rice.edu/~as143/COMP642_Spring22/Scribes/Lect5)  
[OneVSOne / OneVSRest](https://machinelearningmastery.com/one-vs-rest-and-one-vs-one-for-multi-class-classification/)  
[Multi-classifier OneVSOne / OneVSRest](https://towardsdatascience.com/multi-class-classification-one-vs-all-one-vs-one-94daed32a87b)  
[Tutorial Logistic Regression OnevsAll](https://medium.com/analytics-vidhya/logistic-regression-from-scratch-multi-classification-with-onevsall-d5c2acf0c37c)  
[Feature Importance for Logistic Regression](https://forecastegy.com/posts/feature-importance-in-logistic-regression/)  
[Pairplot Interpretation](https://medium.com/analytics-vidhya/pairplot-visualization-16325cd725e6)  
[Explication kurtosis](https://towardsdatascience.com/kurtosis-how-to-how-to-explain-to-a-10-year-old-a3224e615860)  

## Bonus Links

[Wiki stochastic gradient descent](https://fr.wikipedia.org/wiki/Algorithme_du_gradient_stochastique)  
[Explication stochastic gradient descent](https://towardsdatascience.com/stochastic-gradient-descent-clearly-explained-53d239905d31)  
[Difference between stochastic gradient, Batch, Mini Batch](https://towardsdatascience.com/batch-mini-batch-stochastic-gradient-descent-7a62ecba642a)  
[Difference between gradient, stochastic and Mini Batch](https://www.baeldung.com/cs/gradient-stochastic-and-mini-batch)  