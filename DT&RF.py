from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Observing the data, I found two kinds of missing datas
missing_values = ['NaN','?']
data = pd.read_csv('diabetes.csv',na_values=missing_values)
df = pd.DataFrame(data)
df.dropna(axis=0,how='any',inplace=True)

pregnancies = list(df['Pregnancies'])
glucose = list(df['Glucose'])
BloodPressure = list(df['BloodPressure'])
SkinThickness = list(df['SkinThickness'])
Insulin = list(df['Insulin'])
BMI = list(df['BMI'])
DiabetesPedigreeFunction = list(df['DiabetesPedigreeFunction'])
Age = list(df['Age'])
Outcome = list(df['Outcome'])



X = list(zip(pregnancies,glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age))
Y = Outcome

X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size = 0.1,random_state=0)

tree = DecisionTreeClassifier(max_depth=5)
# by comparing the scores of max_depth in range(1,7), 5 seems to be the optimal pruning parameter

tree.fit(X_train,y_train)
score = tree.score(X_test,y_test)
print('the score for decision tree is {}\n'.format(score))

forest = RandomForestClassifier(n_estimators= 40, random_state=42)
forest.fit(X_train,y_train),
scoref = forest.score(X_test,y_test)
print('the score for random forest is {}\n'.format(scoref))

# the score for decision tree is 0.78 while that for random forest is 0.81
# so we use the random forest prediction.

Y_predict = forest.predict(X_test)
names = {'No','Yes'}

for i in range(len(Y_predict)):
    if Y_predict[i] == y_test[i]:
        correct = 1
    else:
        correct = 0
    print('Data: ',X_test[i],'Predicted: ',Y_predict[i],'Actual: ',y_test[i],'Correctness: ',correct)

# We can now use the forest to estimate the relative importance of features:
def plot_feature_importances_cancer(model):
     n_features = 8
     plt.barh(range(n_features), model.feature_importances_, align='center')
     plt.yticks(np.arange(n_features), ['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age'])
     plt.xlabel("Feature importance")
     plt.ylabel("Feature")
plot_feature_importances_cancer(forest)
plt.show()

# from the bar chart, we know that Glucose, BMI and Age are the three top factors for diabetes, according to this model.