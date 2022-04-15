# Random_forest_diabetes
Using Decision trees and Random forests to determine the type of attribute that reflects diabetes the most.

In this project I aim to compare the major features that one may consider when trying to classify potential patients as having diabetes or not. They include times of pregnancies, glucose, BMI, age and so on. In the dataset there are 769 instances and contains missing data so pandas "dropna" need to be used to clean the data. 

Since it is a classification problem, I used it to contrast the effectiveness of a single decision tree and a random forest. It turns out that the random forest algorithm achieves better score in the test data sets (over 0.84) comparing to a decision tree model, which has a maximum score of 0.77 when max_depth is set to be 5.

Taking the seemingly more reliable random_forest model, A bar chart was plotted showing that Glucose, BMI and Age are the three top indicators.

This project was inspired by Kaggle, there website is as below:
https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database
