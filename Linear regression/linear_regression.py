import torch

class LinearRegression():

    def __init__(self,lr:float=0.001,n_iters:int=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.w = None
        self.b = None
    


    def fit(self,X,Y):
        if not isinstance(X, torch.Tensor):
            X = torch.from_numpy(X).float()
        if not isinstance(Y, torch.Tensor):
            Y = torch.from_numpy(Y).float()
        rows, n_features = X.shape
        # #print(X.shape)
        self.w = torch.zero_(torch.Tensor(n_features))
        self.b = torch.zero_(torch.Tensor(1))

        #print(self.w.shape)
        #print(self.b.shape)

        for _ in range(self.n_iters):
            y_pred  = self.w @ X.T + self.b  ## w.Xt +b

            dw = (2*X.T@(y_pred-Y))/rows    ## dw = 2/n * X.T @ (Y - y_pred)
            db = (2*torch.sum((y_pred-Y))/rows)        ## db = 2(y-y_pred)/n

            self.w = self.w - self.lr * dw
            self.b = self.b - self.lr * db

    def predict(self,X):
        if not isinstance(X, torch.Tensor):
            X = torch.from_numpy(X).float()
        #print(self.w.shape)
        #print(X.shape)
        #print(self.b.shape)
        y_pred = self.w @ X.T + self.b
        return y_pred
    
    








    