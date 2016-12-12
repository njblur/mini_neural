import numpy as np
from neural import *

n = network(square_loss,prime_square_loss,learning_rate=0.01)
l = layer(2,2,nop,prime_nop,bias_rate=0.0)
l2 = layer(2,1,nop,prime_nop,bias_rate=0.0)

# n.add_layer(l)
n.add_layer(l2)
x1 = np.array([[1,1]]).reshape(2,-1)
y1 = np.array([30]).reshape(1,-1)
x2 = np.array([[2,4]]).reshape(2,-1)
y2 = np.array([100]).reshape(1,-1)

for _ in range(1500):
    loss1 = n.learn(x2,y2)
    loss2 = n.learn(x1,y1)
    loss = (loss1+loss2)
    n.set_learning_rate(n.learning_rate*0.95)
    if(loss<0.5):
        break
    # loss = n.learn(x,y)



print l2.weights
