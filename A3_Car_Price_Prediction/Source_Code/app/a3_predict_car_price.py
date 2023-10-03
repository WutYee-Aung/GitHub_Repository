import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import pickle
import time

filename = '/root/source_code/app/a3_car_price_prediction.model'

from sklearn.model_selection import KFold
from sklearn.preprocessing import PolynomialFeatures

class LogisticRegression:
    
    def __init__(self, regularization, k, n, method, alpha = 0.001, max_iter=5000):
        self.k = k
        self.n = n
        self.alpha = alpha
        self.max_iter = max_iter
        self.method = method
        self.regularization = regularization
    
    def fit(self, X, Y):
        self.W = np.random.rand(self.n, self.k)
        self.losses = []
        
        if self.method == "batch":
            start_time = time.time()
            for i in range(self.max_iter):
                loss, grad =  self.gradient(X, Y)
                self.losses.append(loss)
                self.W = self.W - self.alpha * grad
                # if i % 500 == 0:
                #     print(f"Loss at iteration {i}", loss)
            print(f"time taken: {time.time() - start_time}")
            
        elif self.method == "minibatch":
            start_time = time.time()
            batch_size = int(0.3 * X.shape[0])
            for i in range(self.max_iter):
                ix = np.random.randint(0, X.shape[0]) #<----with replacement
                batch_X = X[ix:ix+batch_size]
                batch_Y = Y[ix:ix+batch_size]
                loss, grad = self.gradient(batch_X, batch_Y)
                self.losses.append(loss)
                self.W = self.W - self.alpha * grad
                # if i % 500 == 0:
                #     print(f"Loss at iteration {i}", loss)
            print(f"time taken: {time.time() - start_time}")
            
        elif self.method == "sto":
            start_time = time.time()
            list_of_used_ix = []
            for i in range(self.max_iter):
                idx = np.random.randint(X.shape[0])
                while i in list_of_used_ix:
                    idx = np.random.randint(X.shape[0])
                X_train = X[idx, :].reshape(1, -1)
                Y_train = Y[idx]
                loss, grad = self.gradient(X_train, Y_train)
                self.losses.append(loss)
                self.W = self.W - self.alpha * grad
                
                list_of_used_ix.append(i)
                if len(list_of_used_ix) == X.shape[0]:
                    list_of_used_ix = []
                # if i % 500 == 0:
                #     print(f"Loss at iteration {i}", loss)
            print(f"time taken: {time.time() - start_time}")
            
        else:
            raise ValueError('Method must be one of the followings: "batch", "minibatch" or "sto".')
        
        
    def gradient(self, X, Y):
        m = X.shape[0]
        h = self.h_theta(X, self.W)
        loss = (- np.sum(Y*np.log(h)) / m) + self.regularization(self.W)
        error = h - Y
        grad = self.softmax_grad(X, error) + self.regularization.derivation(self.W)
        return loss, grad

    def softmax(self, theta_t_x):
        return np.exp(theta_t_x) / np.sum(np.exp(theta_t_x), axis=1, keepdims=True)

    def softmax_grad(self, X, error):
        return  X.T @ error

    def h_theta(self, X, W):
        '''
        Input:
            X shape: (m, n)
            w shape: (n, k)
        Returns:
            yhat shape: (m, k)
        '''
        return self.softmax(X @ W)
    
    def predict(self, X_test):
        return np.argmax(self.h_theta(X_test, self.W), axis=1)
    
    def plot(self):
        plt.plot(np.arange(len(self.losses)) , self.losses, label = "Train Losses")
        plt.title("Losses")
        plt.xlabel("epoch")
        plt.ylabel("losses")
        plt.legend()
        
    def set_confustion_matrix(self, confusion_matrix):
        self.confusion_matrix = confusion_matrix
    
    def cal_accuracy(self):
        true_pred = np.sum(np.diagonal(self.confusion_matrix))
        total_pred = np.sum(self.confusion_matrix)
        return true_pred / total_pred
    
    def cal_precision(self, class_label):
        true_positives = self.confusion_matrix[class_label, class_label]
        false_positives = np.sum(self.confusion_matrix[:, class_label]) - true_positives
        
        if true_positives + false_positives == 0:
            return 0  # Avoid division by zero
        
        precision_c = true_positives / (true_positives + false_positives)
        return precision_c

    def cal_recall(self, class_label):
        true_positives = self.confusion_matrix[class_label, class_label]
        false_negatives = np.sum(self.confusion_matrix[class_label, :]) - true_positives
        
        if true_positives + false_negatives == 0:
            return 0  # Avoid division by zero
        
        recall_c = true_positives / (true_positives + false_negatives)
        return recall_c

    def cal_f1_score(self, class_label):
        prec = self.cal_precision(class_label)
        rec = self.cal_recall(class_label)
        
        if prec + rec == 0:
            return 0  # Avoid division by zero
        
        f1_c = 2 * (prec * rec) / (prec + rec)
        return f1_c
    
    def macro_precision(self):
        total_precision = 0.0
        for class_label in range(self.k):
            total_precision += self.cal_precision(class_label)
        
        macro_precision_score = total_precision / self.k
        return macro_precision_score

    def macro_recall(self):
        total_recall = 0.0
        for class_label in range(self.k):
            total_recall += self.cal_recall(class_label)
        
        macro_recall_score = total_recall / self.k
        return macro_recall_score

    def macro_f1_score(self):
        total_f1 = 0.0
        for class_label in range(self.k):
            total_f1 += self.cal_f1_score(class_label)
        
        macro_f1_score = total_f1 / self.k
        return macro_f1_score
    
    def set_class_weights(self, class_weight):
        self.class_weight = class_weight

    def weighted_precision(self):
        weighted_precision_score = 0.0
        for class_label in range(self.k):
            precision_c = self.cal_precision(class_label)
            weighted_precision_score += self.class_weight[class_label] * precision_c
        
        return weighted_precision_score

    def weighted_recall(self):
        weighted_recall_score = 0.0
        for class_label in range(self.k):
            recall_c = self.cal_recall(class_label)
            weighted_recall_score += self.class_weight[class_label] * recall_c
        
        return weighted_recall_score

    def weighted_f1_score(self):
        weighted_f1_score = 0.0
        for class_label in range(self.k):
            f1_c = self.cal_f1_score(class_label)
            weighted_f1_score += self.class_weight[class_label] * f1_c
        
        return weighted_f1_score
    
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
    
class NormalPenalty:
    def __init__(self):
        pass
    
    def __call__(self, theta):
        return 0
    
    def derivation(self, theta):
        return 0
    
class Lasso(LogisticRegression):
    def __init__(self, k, n, method, l):
        self.regularization = LassoPenalty(l)
        super().__init__(self.regularization, k, n, method)
        
class Ridge(LogisticRegression):
    def __init__(self, k, n, method, l):
        self.regularization = RidgePenalty(l)
        super().__init__(self.regularization, k, n, method)
        
class Normal(LogisticRegression):
    def __init__(self, k, n, method, l):
        self.regularization = NormalPenalty()
        super().__init__(self.regularization, k, n, method)
        
def fn_a3_predict(to_predict_list):

    to_predict = np.array(to_predict_list).reshape(1,3)

    loaded_model = pickle.load(open(filename, 'rb'))
    model = loaded_model['model']
    scaler = loaded_model['scaler']

    to_predict = scaler.transform(to_predict)

    result = model.predict(to_predict)
    return np.exp(result[0])