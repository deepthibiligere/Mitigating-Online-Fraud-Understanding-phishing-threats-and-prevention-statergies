# for numerical computing
import numpy as np

# for dataframes
import pandas as pd


# Ignore Warnings
import warnings
warnings.filterwarnings("ignore")


# to split train and test set
from sklearn.model_selection import train_test_split


# Machine Learning Models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


from sklearn.metrics import accuracy_score


# to save the final model on disk

data=pd.read_csv("phishcoop.csv")
data = data.drop('id',axis=1) #removing unwanted column
print(data.shape)
print(data.columns)
print(data.head())
print(data.describe())
print(data.corr())
print(data.info())
data = data.drop_duplicates()
print( data.shape )

print(data.isnull().sum())
data=data.dropna()
print(data.isnull().sum())



import seaborn as sns
import matplotlib.pyplot as plt
sns.countplot(data['Result'])
plt.xlabel("-1 = Phishing Website, 1 = Non Phishing Website")
plt.title("Phishing Websites Detection")
plt.show()


#Unique values for each columns
col=data.columns
for i in col:
     if  i!='index':
        print(i,data[i].unique())



plt.figure(figsize=(15, 15))
sns.heatmap(data.corr(), linewidths=.5)
plt.title("Correlation Graph")
plt.show()






        

y = data.Result

# Create separate object for input features
X = data.drop('Result', axis=1)




# Split X and y into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y,  test_size=0.2, random_state=0)

# Print number of observations in X_train, X_test, y_train, and y_test
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

model1= LogisticRegression()

#model2=RandomForestClassifier(n_estimators = 200, criterion = "gini", max_features = 'log2',  random_state=42 )
'''
model2=RandomForestClassifier(random_state = 42,
                                  n_estimators = 750,
                                  max_depth = 15, 
                                  min_samples_split = 5,  min_samples_leaf = 1)
'''                                  
model2=RandomForestClassifier(random_state = 42, max_depth = 15, n_estimators = 200, min_samples_split = 2, min_samples_leaf = 1)
 

model4= KNeighborsClassifier(n_neighbors=7)
model5=DecisionTreeClassifier()
model6= GaussianNB()
model7=SVC()

model1.fit(X_train, y_train)
model2.fit(X_train, y_train)

model4.fit(X_train, y_train)
model5.fit(X_train, y_train)
model6.fit(X_train, y_train)
model7.fit(X_train, y_train)




## Predict Test set results
y_pred1 = model1.predict(X_test)
y_pred2 = model2.predict(X_test)
y_pred4 = model4.predict(X_test)
y_pred5 = model5.predict(X_test)
y_pred6 = model6.predict(X_test)
y_pred7 = model7.predict(X_test)






# Decision Tree model 
from sklearn.tree import DecisionTreeClassifier

# instantiate the model 
tree = DecisionTreeClassifier(max_depth = 5)
# fit the model 
tree.fit(X_train, y_train)

#predicting the target value from the model for the samples
y_test_tree = tree.predict(X_test)
y_train_tree = tree.predict(X_train)


#checking the feature improtance in the model
plt.figure(figsize=(9,7))
n_features = X_train.shape[1]
plt.barh(range(n_features), tree.feature_importances_, align='center')
plt.yticks(np.arange(n_features), X_train.columns)
plt.xlabel("Feature importance")
plt.ylabel("Feature")
plt.show()





acc1 = accuracy_score(y_test, y_pred1) ## get the accuracy on testing data
print("Accuracy of Logistic Regression is {:.2f}%".format(acc1*100))



acc2 = round(accuracy_score(y_test, y_pred2),2)## get the accuracy on testing data
print("Accuracy of RandomForestClassifier is {:.2f}%".format(acc2*100))






acc4 = accuracy_score(y_test, y_pred4) ## get the accuracy on testing data
print("Accuracy of KNeighborsClassifier is {:.2f}%".format(acc4*100))




acc5 = accuracy_score(y_test, y_pred5)  ## get the accuracy on testing data
print("Accuracy of Decision Tree is {:.2f}%".format(acc5*100))




acc6 = accuracy_score(y_test, y_pred6)  ## get the accuracy on testing data
print("Accuracy of GaussianNB is {:.2f}%".format(acc6*100))



acc7 = accuracy_score(y_test, y_pred7)  ## get the accuracy on testing data
print("Accuracy of SVC is {:.2f}%".format(acc7*100))




#for plotting
import matplotlib.pyplot as plt
import seaborn as sns


#from sklearn.externals import joblib 
import joblib

# Save the model as a pickle in a file 
joblib.dump(model2, 'phishing.pkl') 
  
# Load the model from the file 
final_model = joblib.load('phishing.pkl')

pred=final_model.predict(X_test)


acc = round(accuracy_score(y_test,pred),2) ## get the accuracy on testing data
print("Final Model Accuracy is {:.2f}%".format(acc*100))


