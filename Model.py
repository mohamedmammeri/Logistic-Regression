import numpy as np
import matplotlib.pyplot as plt


def reshape_dataset(x): 
    #m is number of data
    
    m=x.shape[0]
    x_flatten=x.reshape(m,-1)
    
    return x_flatten,m
#standardize dataset for image data    
#def standardize_dataset(x):
#   return x=x/255

# sigmoid function
def sigmoid(z):
    s=1/(1+np.exp(-z))
    return s
    
#initialize whight with zeros
def init_zeros(dim):
    w=np.zeros((dim,1),dtype=float)
    b=float(0)
    return w,b

#forward propagation
def forward(w,b,x,y,m):
    
    y_hat=sigmoid((np.dot(w.T,x)+b))
    
    cost_func=np.sum((Y*np.log(y_hat))+((1-Y)*np.log(1-y_hat)))
    cost_func/=m
    
    #calcul derivative
    dw=np.dot(X,(y_hat-Y).T)
    dw/=m
    db=np.sum(y_hat-Y)
    db/=m
    
    cost = np.squeeze(np.array(cost_func))
    
    return dw,db,cost

def train(w, b, x, y,iteration,alpha,m):
    
    for i in range(iteration):
        dw,db,cost=forward(w,b,x,y,m)
        
        #correct the whight
        w=w-(alpha*dw)
        b=b-(alpha*db)
        
       
    return w,b,cost
        
    
def fit(x_train,y_train,iterations,alpha): 

    # reshape training Matrix to one colomn for each training data
    x_train,m=reshape_dataset(x_train)
    
    w,b=init_zeros(m)
    
    w,b,cost=train(w,b,x_train,y_train,iterations,alpha,m)
    return w,b,cost
    
    
X = np.array([[1., -2., -1.], [3., 0.5, -3.2]])
Y = np.array([[1, 1, 0]])

w,b,cost=fit(X,Y,100000,1)
print(w)

print(b)
