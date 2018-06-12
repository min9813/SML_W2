
# coding: utf-8

# In[126]:


import numpy as np 
import matplotlib.pyplot as plt
import math

get_ipython().magic('matplotlib inline')

PI = math.pi

def homework1_random(n):
    x_data = np.linspace(0,5,n)
    x_data_1 = x_data[x_data<=2]
    x_data_2 = x_data[(2<x_data)&(x_data<=3)]
    x_data_3 = x_data[(3<x_data)&(x_data<=5)]
    y_data_1 = (1-np.absolute(x_data_1-1))/4
    y_data_2 = np.full(x_data_2.shape,1/4)
    y_data_3 = (1-np.absolute(x_data_3-4))/2
    y_data = np.hstack([y_data_1,y_data_2,y_data_3])
    return x_data,y_data

def homework2_random(n):
    """
        create n samples of probabial valiavles
    """
    x_data = np.zeros(n)
    u = np.random.rand(n)
    flag = (0<u)&(u<1/8)
    x_data[flag] = np.sqrt(8*u[flag])
    flag = (1/8<=u)&(u<1/4)
    x_data[flag] = 2-np.sqrt(2-8*u[flag])
    flag = (1/4<=u)&(u<1/2)
    x_data[flag] = 1+4*u[flag]
    flag = (1/2<=u)&(u<3/4)
    x_data[flag] = 3+np.sqrt(4*u[flag]-2)
    flag = (3/4<=u)&(u<=1)
    x_data[flag] = 5-np.sqrt(4-4*u[flag])
    return u,x_data

def homework_gauss(x,d):
    if d==1:
        y = math.exp(-1/2*x**2)/((2*PI)**1/2)
    else:
        y = np.exp(-1/2*np.dot(x.reshape(d,-1),x.reshape(-1,d)))/((2*PI)**d/2)
    return y

def normalize(a,x,h):
    z = (a-x)/(h+1e-5)
    return z

def prob_function(a,x,h,d):
    if d ==1:
        base_array = 0
    else:
        base_array = np.zeros((d,1))
    normals = normalize(a,x,h)
    for normal in normals:
        base_array += homework_gauss(normal,d)
    return base_array

def cross_validate(data, sample, h, d,cv=5):
    one_samples = int(sample/cv)
    result = 0
    for k in range(cv):
        test = data[k*one_samples:(k+1)*one_samples] 
        if k==0:
            train = data[(k+1)*one_samples:]
        elif k==cv-1:
            train = data[:k*one_samples]
        else:
            train = np.hstack((data[0:k*one_samples],data[(k+1)*one_samples:]))
        value = sample*(h**d)
        y = np.array([prob_function(i,train,h, d) for i in test])/value#aaaaaaaa
        lcv = np.mean(np.log(y)+1e-5)
        if k == 0:
            result = lcv
        else:
            result += lcv
    result = result/cv
    return result
        


# In[43]:


x,y=homework2_random(1000)
y.shape
plt.hist(y,bins=10)


# In[127]:


sample = 1000
d=1
x,x_samples = homework2_random(sample)


# In[129]:


h = np.linspace(0,0.5,30)
r = [cross_validate(x_samples, sample, height, d) for height in h]

plt.plot(h,r)

