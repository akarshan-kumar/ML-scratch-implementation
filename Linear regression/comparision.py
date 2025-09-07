from sklearn import datasets
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from linear_regression import LinearRegression as ScratchLinearRegression

X,Y = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=42)
x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size=0.2, random_state=42)

fig = plt.figure(figsize=(8,6))
plt.scatter(X[:,0],Y)

model1 = LinearRegression()
model1.fit(x_train,y_train)

predictions1 = model1.predict(X)
error1 = mean_squared_error(Y,predictions1)

print(error1)

cmap = plt.get_cmap('viridis')
m1 = plt.scatter(X[:,0],predictions1, color=cmap(0.9), marker='.', label='Sklearn Model')
m2 = plt.scatter(X[:,0],Y, color=cmap(0.5), marker='.', label='Data')
# plt.legend(handles=[m1,m2])


model2 = ScratchLinearRegression()
model2.fit(x_train,y_train)

predictions2 = model2.predict(X)
error2 = mean_squared_error(Y,predictions2)

print(error2)

cmap = plt.get_cmap('viridis')
m1 = plt.scatter(X[:,0],predictions2, color=cmap(0.2), marker='.', label='Scratch Model')
# m2 = plt.scatter(X[:,0],Y, color=cmap(0.5), marker='.', label='Data')
# plt.legend(handles=[m1,m2])
plt.show()
