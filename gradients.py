import numpy as np

X = np.array([1,2,3,4], dtype=np.float32)
Y = np.array([2,4,6,8], dtype=np.float32)

w = 0.0

def forward(x):
    return w*X

def loss(y,y_predicted):
    return ((y-y_predicted)**2).mean()


# calculating gradient in terms of x
def gradient(x,y,y_predicted):
    return np.dot(2*x,y_predicted-y).mean()

learning_rate = 0.001
n_iter = 40

for epoch in range(n_iter):
    # prediction
    y_predicted = forward(X)
    # loss
    l = loss(Y,y_predicted)
    # gradient
    dw = gradient(X,Y,y_predicted)
    #update weights
    w -= learning_rate*dw

    if(epoch%1==0):
        print(f'epoch {epoch+1}: w = {w:.3f}, loss = {l:.8f}')

print(forward(5))
