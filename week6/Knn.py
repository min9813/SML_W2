# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd


PI = math.pi
class Kneighbor():
    def __init__(self,n_neighbors=1):
        self.n_neighbors = 1
        self.x_data = None
        self.y_data = None
        self.dim = None
        self.class_label = None

    def fit(self,x_train, y_train):
        print("x_train:",x_train.shape)
        print("y_train:",y_train.shape)
        if len(x_train.shape) > 1:
            print("yes")
            self.dim = x_train.shape[1]
        else:
            self.dim = 1
        print("dim:",self.dim)
        self.class_label = np.unique(y_train)
        self.x_data = x_train.reshape(-1,self.dim)
        self.y_data = y_train.reshape(-1,1)


    def predict(self, x_test):
        result = np.array([self.get_predict(x,v) for v,x in enumerate(x_test)])
        return result

    def get_predict(self, x_test,v):
        if v%50 == 0:
            print("v={}".format(v))
        distance = np.dot((self.x_data-x_test),(self.x_data-x_test).T)
        candidate_id = np.argsort(distance)[::-1][:self.n_neighbors]
        candidate = self.y_data
        count_class = [np.sum(candidate==cls)for cls in self.class_label]
        result_id = np.argmax(count_class)
        _cls = self.class_label[result_id]
        return _cls


def cross_validate(x_data, y_data, n_neighbor,cv=5):
    if len(x_data.shape) > 1:
        sample = x_data.shape[0]
    else:
        sample = x_data.shape[0]
        x_data = x_data.reshape(sample,1)
    one_samples = int(sample/cv)
    result = 0
    for k in range(cv):
        x_test = x_data[k*one_samples:(k+1)*one_samples]
        y_test = y_data[k*one_samples:(k+1)*one_samples]
        if k==0:
            x_train = x_data[(k+1)*one_samples:]
            y_train = y_data[(k+1)*one_samples:]
        elif k==cv-1:
            x_train = x_data[:k*one_samples]
            y_train = y_data[:k*one_samples]
        else:
            x_train = np.vstack((x_data[0:k*one_samples],x_data[(k+1)*one_samples:]))
            y_train = np.vstack((y_data[0:k*one_samples],y_data[(k+1)*one_samples:]))
        cls = Kneighbor()
        cls.fit(x_train, y_train)
        p = cls.predict(x_test)
        print(p)
        accuracy = np.mean((p==y))
        result += accuracy
    result = result/cv
    return result

def load_data():
    list_data = list(range(10))
    full_matrix = None
    digit_test = [pd.read_csv("digit_test{}.csv".format(str(n)),header=None).values for n in list_data]
    digit_train = [pd.read_csv("digit_train{}.csv".format(str(n)),header=None).values for n in list_data]
    y_true = np.array([0]*500+[1]*500+[2]*500+[3]*500+[4]*500+[5]*500+[6]*500+[7]*500+[8]*500+[9]*500)
    whole_train = np.vstack(digit_train)
    print(y_true.shape)
    train_with_answer = np.hstack((whole_train, y_true.reshape(-1,1)))
    return train_with_answer, digit_test


train_data, test_data = load_data()

accuracy_list = []
x_data = train_data[:,:-1]
y_data = train_data[:,-1]
cross_validate(x_data, y_data, 3)
