from sklearn import datasets
from sklearn.metrics import f1_score,precision_score,recall_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from Log_reg import LogisticRegressionModel as ScratchLogisticRegression    

data = datasets.load_breast_cancer()

x_train,x_test,y_train,y_test = train_test_split(data.data,data.target,test_size=0.2,random_state=42)

model1 = LogisticRegression()
model1.fit(x_train,y_train)

predictions1 = model1.predict(x_test)
precision1 = precision_score(y_test,predictions1)
recall1 = recall_score(y_test,predictions1)
f1_1 = f1_score(y_test,predictions1)

print(precision1)
print(recall1)  
print(f1_1)

model2 = ScratchLogisticRegression(lr=0.01,n_iter=10000)
model2.fit(x_train,y_train) 
predictions2 = model2.predict(x_test)
precision2 = precision_score(y_test,predictions2)
recall2 = recall_score(y_test,predictions2)
f1_2 = f1_score(y_test,predictions2)

print(precision2)
print(recall2)
print(f1_2)
