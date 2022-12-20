# Ml--matrixmethods-
Categorical Naive Bayes , Gaussian matrix , neural network and SVM 
This module has the application of categorical naive bayes , gaussian, neural networking and SVM methods. 

## Categorical Naive Bayes 
Naïve bayes algorithm is a supervised learning algorithm based on applying Bayes theorem. 
The standard basic assumption for Naïve’s method is assuming all variables independent. Classification or categorical methods are dominantly meant to group the data into yes or no questions for example 
Probability of an email being either a spam or not a spam 
Probability of a person having cancer or not 
In the assumption of every feature of data being independent for most of the other features it uses bayes formula (as shown in fig ) to allocate probabilities for features . 

## Gausian Matrix Model 
Gaussian Mixture Model is used to present probabilistic model using normal distribution and finding subpopulations in the overall population . As the subpopulation isn’t pre-determined  and the model makes its own subpopulations from the existing data points , mixture model is known as unsupervised learning . 
Gaussian Mixture Model can moderate any kind of data i.e univariate, multivariate .  

Under the hood, a Gaussian mixture model is very similar to k-means: it uses an expectation–maximization approach which qualitatively does the following:
Choose starting guesses for the location and shape
Repeat until converged:
•	E-step: for each point, find weights encoding the probability of membership in each cluster
•	M-step: for each cluster, update its location, normalization, and shape based on all data points, making use of the weights
The result of this is that each cluster is associated not with a hard-edged sphere, but with a smooth Gaussian model. Just as in the k-means expectation–maximization approach, this algorithm can sometimes miss the globally optimal solution, and thus in practice multiple random initializations are used.
Expectation maximization (EM) is seemingly the most popular technique used to determine the parameters of a mixture with an a priori given number of components. This is a particular way of implementing maximum likelihood estimation for this problem.

## Backpropagation Neural Network 
Backpropagation algorithm is a supervised learning method for Multilayer Feed Forward Network. Neural Network is usually a multilayer system of input values , hidden layers and output layers . It tends to take in input values , break them down through the hidden layer weights and help in prediction of the possible out puts . This is mostly used in image recognition and pixel reading . While constructing a neural network system , we usually consider the depth of the neural network . 

The core concept of Backpropagation Neural Network(BPN) is to back propagate the error term . This helps in  spreading  the  error from the output layer to internal hidden layer in order to retune the weights and decrease the error rates . It helps finetune the weights at each iteration. 

Backpropagation follows three basic steps 
-	Initializing network 
-	Forward propagation 
-	Back propagation error 

## Support Vector Machine 
Support Vector Machine is a special linear classifier that is usually used for supervised learning that are used for regression models . Given a set of training examples, each marked as belonging to one of two categories, an SVM training algorithm builds a model that assigns new examples to one category or the other, making it a non-probabilistic binary linear classifier (although methods such as Platt scaling exist to use SVM in a probabilistic classification setting). 

 
Support Vector machine is a hyperplane based classifier defined by w and b . We begin the process with the assumption that the data is linearly separated.

Support vectors are usually the values that lie closest to the hyperplane . There are usually many hyperplanes for a data set to separate the dataset . The best hyper plane is the one that has the maximum separation between the two classes . This method can not only be used for classification but also regression and identifying outliner values. 

In SVM, we take the output of the linear function and if that output is greater than 1, we identify it with one class and if the output is -1, we identify is with another class. Since the threshold values are changed to 1 and -1 in SVM, we obtain this reinforcement range of values([-1,1]) which acts as margin.

## Code 
Coding for most of the prediction models are divided into the below steps 
- Preprocessing the data and removing the redundant values 
- dividing the data set into training and testing based on what we need to predict 
- Defining the predictive model on the training data set 
- applying the predictor to the test data set 
- error understanding 

## Conclusion 
The predicted value was mapped against the true value to understand how many of the values actually aligned . Based on the confusion matrix the true positives ( blue line ) is the highest . Plotting the same on a graph the outputs are as below . 

## References 
https://en.wikipedia.org/wiki/Support_vector_machine 
Prof.Changyu Chen Slides 
Data from kaggle 
Libraries https://scikitlearn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
error clearing code – stack overflow 
https://towardsdatascience.com/support-vector-machine-introduction-to-machine-learning-algorithms934a444fca47
https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/
data from : UCI Machine Learning 
Libraries numpy , random , pandas , matplotlib etc 
error clearing code – stack overflow 
https://pyimagesearch.com/2021/05/06/backpropagation-from-scratch-with-python/
https://github.com/cchangyou/simple-neural-network/blob/master/neural-network.py




 
 
