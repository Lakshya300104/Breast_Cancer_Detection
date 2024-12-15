import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import pickle



data = pd.read_csv('data.csv')
data = data.drop(columns='id', axis=1)
X = data.drop(columns='label', axis=1)
Y = data['label']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)
model = LogisticRegression(max_iter = 1000)
model.fit(X_train, Y_train)
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(Y_train, X_train_prediction)
print('Accuracy on training data LR = ', training_data_accuracy)  #95.8% accuracy
# accuracy on test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(Y_test, X_test_prediction)
print('Accuracy on test data LR = ', test_data_accuracy) #93.85 % accuracy
filename = 'trained_modelLR.sav'
file = open(filename,'wb')
pickle.dump(model, file)

model2 = MultinomialNB()
model2.fit(X_train, Y_train)
X_train_prediction = model2.predict(X_train)
training_data_accuracy = accuracy_score(Y_train, X_train_prediction)
print('Accuracy on training data NB = ', training_data_accuracy)  #9.1% accuracy
# accuracy on test data
X_test_prediction = model2.predict(X_test)
test_data_accuracy = accuracy_score(Y_test, X_test_prediction)
print('Accuracy on test data = NB ', test_data_accuracy) #87.7 % accuracy
filename = 'trained_modelNB.sav'
file2 = open(filename,'wb')
pickle.dump(model2, file2)
