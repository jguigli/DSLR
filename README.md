# DSLR
Data Science Logistic Regression  

Useful links :  
[SUBJECT](https://cdn.intra.42.fr/pdf/pdf/66152/en.subject.pdf)  
[PANDAS GETTING STARTED](https://pandas.pydata.org/docs/getting_started/index.html#getting-started)  
[Multi-classifier Logistic Regression](https://www.cs.rice.edu/~as143/COMP642_Spring22/Scribes/Lect5)  
[OneVSOne / OneVSRest](https://machinelearningmastery.com/one-vs-rest-and-one-vs-one-for-multi-class-classification/)  
[Multi-classifier OneVSOne / OneVSRest](https://towardsdatascience.com/multi-class-classification-one-vs-all-one-vs-one-94daed32a87b)  
[Tutorial Logistic Regression OnevsAll](https://medium.com/analytics-vidhya/logistic-regression-from-scratch-multi-classification-with-onevsall-d5c2acf0c37c)  
[Feature Importance for Logistic Regression](https://forecastegy.com/posts/feature-importance-in-logistic-regression/)  
[Pairplot Interpretation](https://medium.com/analytics-vidhya/pairplot-visualization-16325cd725e6)  
[Explication kurtosis](https://towardsdatascience.com/kurtosis-how-to-how-to-explain-to-a-10-year-old-a3224e615860)  

Links for bonus :  
[Wiki stochastic gradient descent](https://fr.wikipedia.org/wiki/Algorithme_du_gradient_stochastique)  
[Explication stochastic gradient descent](https://towardsdatascience.com/stochastic-gradient-descent-clearly-explained-53d239905d31)  
[Difference between stochastic gradient, Batch, Mini Batch](https://towardsdatascience.com/batch-mini-batch-stochastic-gradient-descent-7a62ecba642a)  
[Difference between gradient, stochastic and Mini Batch](https://www.baeldung.com/cs/gradient-stochastic-and-mini-batch)  

It is highly recommended to perform the steps in following order.  

## Data Analysis  

describe.py : `Display information for all numerical features in dataset.csv`  

	        Feature 1   Feature 2   Feature 3   Feature 4
	Count 149.000000    149.000000  149.000000  149.000000
	Mean    5.848322    3.051007    3.774497    1.205369
	Std     5.906338    3.081445    4.162021    1.424286
	Min     4.300000    2.000000    1.000000    0.100000
	25%     5.100000    2.800000    1.600000    0.300000
	50%     5.800000    3.000000    4.400000    1.300000
	75%     6.400000    3.300000    5.100000    1.800000
	Max     7.900000    4.400000    6.900000    2.500000

## Data visualization  

### Histogram  
Make a script called histogram.[extension] which displays a histogram answering the
next question :  

    Which Hogwarts course has a homogeneous score distribution between all four houses?

### Scatter plot  

Make a script called scatter_plot.[extension] which displays a scatter plot answering  
the next question :  

    What are the two features that are similar ?

### Pair plot  

Make a script called pair_plot.[extension] which displays a pair plot or scatter plot  
matrix (according to the library that you are using).  

    From this visualization, what features are you going to use for your logistic regression?

## Logistic Regression  

You arrive at the last part: code your Magic Hat. To do this, you have to perform a
multi-classifier using a logistic regression one-vs-all.  

You will have to make two programs :  

• First one will train your models, it’s called logreg_train.[extension]. It takes  
as a parameter dataset_train.csv. For the mandatory part, you must use the  
technique of gradient descent to minimize the error. The program generates a file  
containing the weights that will be used for the prediction.  

• A second has to be named logreg_predict.[extension]. It takes as a parameter  
dataset_test.csv and a file containing the weights trained by previous program.  
In order to evaluate the performance of your classifier this second program will have  
to generate a prediction file houses.csv formatted exactly as follows:  

    $> cat houses.csv
    Index,Hogwarts House
    0,Gryffindor
    1,Hufflepuff
    2,Ravenclaw
    3,Hufflepuff
    4,Slytherin
    5,Ravenclaw
    6,Hufflepuf

## Bonus Part  

• Add more fields for describe.[extension]  
• Implement a stochastic gradient descent  
• Implement other optimization algorithms (Batch GD/mini-batch GD/ you name it)  

## Correction

Your classifier will be evaluated on the data present in dataset_test.csv.  
Your answers will be evaluated using accuracy score of the `Scikit-Learn` library.  
Professor McGonagall agrees that your algorithm is comparable to the Sorting Hat only if it has a
minimum precision of 98%.
It will also be important to be able to explain the functioning of the used machine learning algorithms.

## Features for logistic regression

'Index', 'Hogwarts House', 'First Name', 'Last Name', 'Birthday', 'Best Hand', 'Arithmancy', 'Astronomy', 'Herbology', 'Defense Against the Dark Arts', 'Divination', 'Muggle Studies', 'Ancient Runes', 'History of Magic', 'Transfiguration', 'Potions', 'Care of Magical Creatures', 'Charms', 'Flying'

Accuracy score `98.75%` with drop features :

    Arithmancy
    Defense Against the Dark Arts
    Transfiguration
    Potions
    Care of Magical Creatures


## Algortihm

Batch Gradient Descent: 
    
    Parameters are updated after computing the gradient of the error with respect to the entire training set
    => Smooth updates in the model parameters

Stochastic Gradient Descent: 

    Parameters are updated after computing the gradient of the error with respect to a single training example
    => Very noisy updates in the parameters

Mini-Batch Gradient Descent: 
    
    Parameters are updated after computing the gradient of  the error with respect to a subset of the training set
    => Depending upon the batch size, the updates can be made less noisy – greater the batch size less noisy is the update

## Bonus fields for describe.py

Mode:

    The mode is the value that appears most frequently in the dataset.

Skewness : 

    Skewness measures the asymmetry of the data distribution relative to the mean. It can be helpful in identifying asymmetric distributions
    => positive = more high values than low values relative to the mean
    => negative = more low values than high values relative to the mean

Kurtosis : 

    Kurtosis measures the shape of the data distribution relative to the normal distribution. It can reveal heavy or light tails compared to the normal distribution
    => positive (>3) = the distribution has thicker tails (more extreme values) and a sharper peak than the normal distribution
    => negative (<3) = the distribution has thinner tails (fewer extreme values) and a less sharp peak than the normal distribution
    => neutral (=3) = corresponds to a normal distribution