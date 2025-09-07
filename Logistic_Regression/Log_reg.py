import torch


class LogisticRegressionModel():

    def __init__(self, lr=0.001,n_iter =1000):
        self.lr = lr
        self.n_iter = n_iter
        self.weights = None
        self.bias = None

    def sigmoid(self,X):
        return 1/(1+torch.exp(-X))
    
    def fit(self, X, Y):
       
        if not isinstance(X, torch.Tensor):
            X = torch.from_numpy(X).float()
        if not isinstance(Y, torch.Tensor):
            Y = torch.from_numpy(Y).float()
        samples, n_features = X.shape
        self.weights = torch.rand(n_features)
        self.bias = torch.rand(1)


        for _ in range(self.n_iter):
            linear_pred = self.weights @ X.T + self.bias
            prob_pred = self.sigmoid(linear_pred)

            dw = (1/samples)*(2*X.T@(prob_pred-Y))
            db = (1/samples)*(2*torch.sum(prob_pred-Y))

            self.weights = self.weights - self.lr*dw
            self.bias = self.bias - self.lr*db


    def predict(self,X):
        if not isinstance(X, torch.Tensor):
            X = torch.from_numpy(X).float()
        linear_pred = self.weights @ X.T + self.bias
        prob_pred = self.sigmoid(linear_pred)

        prob_pred = [0 if i<0.5 else 1 for i in prob_pred]

        return prob_pred
        

        