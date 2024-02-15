# DSLR
Data Science Logistic Regression  

Useful links :  
[SUBJECT](https://cdn.intra.42.fr/pdf/pdf/66152/en.subject.pdf)  
[PANDAS GETTING STARTED](https://pandas.pydata.org/docs/getting_started/index.html#getting-started)  
[Multi-classifier Logistic Regression](https://www.cs.rice.edu/~as143/COMP642_Spring22/Scribes/Lect5)  
[OneVSOne / OneVSRest](https://machinelearningmastery.com/one-vs-rest-and-one-vs-one-for-multi-class-classification/)  
[Multi-classifier OneVSOne / OneVSRest](https://towardsdatascience.com/multi-class-classification-one-vs-all-one-vs-one-94daed32a87b)  
[Tutorial Logistic Regression OnevsAll](https://medium.com/analytics-vidhya/logistic-regression-from-scratch-multi-classification-with-onevsall-d5c2acf0c37c)  

Links for bonus :
[Wiki stochastic gradient descent](https://fr.wikipedia.org/wiki/Algorithme_du_gradient_stochastique)  
[Explication stochastic gradient descent](https://towardsdatascience.com/stochastic-gradient-descent-clearly-explained-53d239905d31)  

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

## REMINDER

OK :
- Valeur pas tout a fait egal pour describe.py
- Faire la partie prediction
- Exporter les thetas dans un csv

KO :
- Describe.py -> rajouter .dropna() ?
- Rajouter les 4 maisons dans le scatter plot avec les couleurs
- Exposer les bonnes features pour la logreg
- Tester le performance du modele
- Valeur NaN remove, ca pose probleme pour le bon nombre de lignes predites dans train et/ou predict
- Rajouter la fonction de cout et la plot
- Faire les bonus
