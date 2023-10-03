import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import matplotlib
import pickle

filename = '/root/source_code/app/a2_car_price_prediction.model'
# C:\AIT\FirstSem\ML\Assignment\A2\A2_Car_Price_Prediction\Source_Code\a2_app\a2_car_price_prediction.model

from sklearn.model_selection import KFold
from sklearn.preprocessing import PolynomialFeatures

class LinearRegression(object):
    
    #in this class, we add cross validation as well for some spicy code....
    kfold = KFold(n_splits=3)
    
    def __init__(self, regularization, init_weights, use_momentum, momentum, degree, lr=0.001, method='batch', num_epochs=500, batch_size=50, cv=kfold):
        
        self.lr             = lr
        self.num_epochs     = num_epochs
        self.batch_size     = batch_size
        self.method         = method
        self.cv             = cv
        self.regularization = regularization
        self.init_weights   = init_weights
        self.use_momentum   = use_momentum
        self.momentum       = momentum
        self.prev_step      = 0.0
        self.degree         = degree  # Degree for polynomial features
        self.poly           = PolynomialFeatures(degree=degree, include_bias=False)  # Initialize PolynomialFeatures

    def mse(self, ytrue, ypred):
        return ((ypred - ytrue) ** 2).sum() / ytrue.shape[0]
    
    def r2(self, ytrue, ypred):
        rss = sum((ytrue - ypred) ** 2)
        y_mean = sum(ytrue) / len(ytrue)
        tss = sum((ytrue - y_mean) ** 2)
        r2 = 1 - (rss / tss)
        return r2
    
    # def fit(self, X_train, y_train):
        
        # Check if polynomial features should be applied
        if self.degree > 1:
            X_train = self.poly.fit_transform(X_train)
            
        #create a list of kfold scores
        self.kfold_scores = list()
        
        #reset val loss
        self.val_loss_old = np.infty

        #kfold.split in the sklearn.....
        #5 splits
        for fold, (train_idx, val_idx) in enumerate(self.cv.split(X_train)):
            
            X_cross_train = X_train[train_idx]
            y_cross_train = y_train[train_idx]
            X_cross_val   = X_train[val_idx]
            y_cross_val   = y_train[val_idx]
            
            if self.init_weights == 'zeros':
                self.theta = np.zeros(X_cross_train.shape[1])
            elif self.init_weights == 'xavier':
                xavier_stddev = np.sqrt(2.0 / (X_cross_train.shape[1] + 1))
                self.theta = np.random.normal(0, xavier_stddev, X_cross_train.shape[1])
            else:
                raise ValueError("Invalid weight initialization method: " + self.init_weights)
        
            # self.theta = np.zeros(X_cross_train.shape[1])
            
            #define X_cross_train as only a subset of the data
            #how big is this subset?  => mini-batch size ==> 50
            
            #one epoch will exhaust the WHOLE training set
            with mlflow.start_run(run_name=f"Fold-{fold}", nested=True):
                
                params = {"method": self.method, "lr": self.lr, "reg": type(self).__name__}
                mlflow.log_params(params=params)
                
                for epoch in range(self.num_epochs):
                
                    #with replacement or no replacement
                    #with replacement means just randomize
                    #with no replacement means 0:50, 51:100, 101:150, ......300:323
                    #shuffle your index
                    perm = np.random.permutation(X_cross_train.shape[0])
                            
                    X_cross_train = X_cross_train[perm]
                    y_cross_train = y_cross_train[perm]
                    
                    if self.method == 'sto':
                        for batch_idx in range(X_cross_train.shape[0]):
                            X_method_train = X_cross_train[batch_idx].reshape(1, -1) #(11,) ==> (1, 11) ==> (m, n)
                            y_method_train = y_cross_train[batch_idx] 
                            train_loss = self._train(X_method_train, y_method_train)
                    elif self.method == 'mini':
                        for batch_idx in range(0, X_cross_train.shape[0], self.batch_size):
                            #batch_idx = 0, 50, 100, 150
                            X_method_train = X_cross_train[batch_idx:batch_idx+self.batch_size, :]
                            y_method_train = y_cross_train[batch_idx:batch_idx+self.batch_size]
                            train_loss = self._train(X_method_train, y_method_train)
                    else:
                        X_method_train = X_cross_train
                        y_method_train = y_cross_train
                        train_loss = self._train(X_method_train, y_method_train)

                    mlflow.log_metric(key="train_loss", value=train_loss, step=epoch)

                    yhat_val = self.predict(X_cross_val)
                    val_loss_new = self.mse(y_cross_val, yhat_val)
                    mlflow.log_metric(key="val_loss", value=val_loss_new, step=epoch)
                    
                    #record dataset
                    mlflow_train_data = mlflow.data.from_numpy(features=X_method_train, targets=y_method_train)
                    mlflow.log_input(mlflow_train_data, context="training")
                    
                    mlflow_val_data = mlflow.data.from_numpy(features=X_cross_val, targets=y_cross_val)
                    mlflow.log_input(mlflow_val_data, context="validation")
                    
                    #early stopping
                    if np.allclose(val_loss_new, self.val_loss_old):
                        break
                    self.val_loss_old = val_loss_new
            
                self.kfold_scores.append(val_loss_new)
                print(f"Fold {fold}: {val_loss_new}")
                    
    # def _train(self, X, y):
        yhat = self.predict(X)
        m    = X.shape[0]        
        grad = (1/m) * X.T @(yhat - y) + self.regularization.derivation(self.theta)
        step = self.lr * grad
        
        if self.use_momentum == 'Yes':
            step += self.momentum * self.prev_step
        
        self.theta = self.theta - step
        self.prev_step = step
        return self.mse(y, yhat)
    
    def predict(self, X):
        return X @ self.theta  #===>(m, n) @ (n, )
    
    def _coef(self):
        return self.theta[1:]  #remind that theta is (w0, w1, w2, w3, w4.....wn)
                               #w0 is the bias or the intercept
                               #w1....wn are the weights / coefficients / theta
    def _bias(self):
        return self.theta[0]
    
    def plot_feature_importance(self, feature_names):
        if not hasattr(self, 'theta'):
            raise ValueError("Model has not been trained yet. Call the 'fit' method first.")

        # Ensure feature_names matches the number of coefficients
        if len(feature_names) != len(self.theta) - 1:
            raise ValueError("Number of feature names should match the number of coefficients.")

        # Exclude the bias term (intercept)
        coefficients = self.theta[1:]

        # Plot the feature importance
        plt.figure(figsize=(10, 6))
        plt.barh(range(len(coefficients)), coefficients, align='center')
        plt.yticks(range(len(coefficients)), feature_names)
        plt.xlabel('Coefficient Value')
        plt.title('Feature Importance')
        plt.show()

class LassoPenalty:
    
    def __init__(self, l):
        self.l = l # lambda value
        
    def __call__(self, theta): #__call__ allows us to call class as method
        return self.l * np.sum(np.abs(theta))
        
    def derivation(self, theta):
        return self.l * np.sign(theta)
    
class RidgePenalty:
    
    def __init__(self, l):
        self.l = l
        
    def __call__(self, theta): #__call__ allows us to call class as method
        return self.l * np.sum(np.square(theta))
        
    def derivation(self, theta):
        return self.l * 2 * theta
    
class ElasticPenalty:
    
    def __init__(self, l = 0.1, l_ratio = 0.5):
        self.l = l 
        self.l_ratio = l_ratio

    def __call__(self, theta):  #__call__ allows us to call class as method
        l1_contribution = self.l_ratio * self.l * np.sum(np.abs(theta))
        l2_contribution = (1 - self.l_ratio) * self.l * 0.5 * np.sum(np.square(theta))
        return (l1_contribution + l2_contribution)

    def derivation(self, theta):
        l1_derivation = self.l * self.l_ratio * np.sign(theta)
        l2_derivation = self.l * (1 - self.l_ratio) * theta
        return (l1_derivation + l2_derivation)
    
class Lasso(LinearRegression):
    
    def __init__(self, method, init_weights, use_momentum, momentum, degree, lr, l):
        self.regularization = LassoPenalty(l)
        super().__init__(self.regularization, init_weights, use_momentum, momentum, degree, lr, method)
        
class Ridge(LinearRegression):
    
    def __init__(self, method, init_weights, use_momentum, momentum, degree, lr, l):
        self.regularization = RidgePenalty(l)
        super().__init__(self.regularization, init_weights, use_momentum, momentum, degree, lr, method)
        
class ElasticNet(LinearRegression):
    
    def __init__(self, method, init_weights, use_momentum, momentum, degree, lr, l, l_ratio=0.5):
        self.regularization = ElasticPenalty(l, l_ratio)
        # super().__init__(self.regularization, lr, method)
        super().__init__(self.regularization, init_weights, use_momentum, momentum, degree, lr, method)
        
def fn_a2_predict(to_predict_list):

    to_predict = np.array(to_predict_list).reshape(1,3)

    loaded_model = pickle.load(open(filename, 'rb'))
    model = loaded_model['model']
    scaler = loaded_model['scaler']

    to_predict = scaler.transform(to_predict)

    result = model.predict(to_predict)
    return np.exp(result[0])