import numpy as np

def nop(x):
    return x
def prime_nop(x):
    return np.ones(x.shape)
def sigmoid(x):
    return 1/(1+np.exp(-x))
def prime_sigmoid(x):
    return sigmoid(x)*(1-sigmoid(x))

class layer:
    def __init__(self,input_size,output_size,fn_activate,fn_prime_activate):
        self.fn_activate = fn_activate
        self.fn_prime_activate = fn_prime_activate
        self.weights = np.zeros((output_size,input_size))
        self.d_weights = np.zeros((output_size,input_size))
        self.bias = np.zeros((output_size,1))
        self.d_bias = np.zeros((output_size,1))
    def forward(self,x):
        self.x = x
        self.z = np.matmul(self.weights,x)+self.bias
        self.out = self.fn_activate(self.z)
        return self.out
    def backward(self,y):
        self.dy = self.fn_prime_activate(self.out) * y
        self.d_weights = np.matmul(self.dy,self.x.T)
        self.d_bias = self.dy
        self.dx = np.matmul(self.weights.T,self.dy)
       
        return self.dx
    def apply_gradients(self,learning_rate):
        self.weights -= self.d_weights*learning_rate
        self.bias -= self.d_bias

def square_loss(out,y):
    l = (out-y)*(out-y)/2
    return np.prod(l)
def prime_square_loss(out,y):
    return (out-y)

class network:
    def __init__(self,loss_func,prime_loss_func,learning_rate=0.001):
        self.loss_func = loss_func
        self.prime_loss_func = prime_loss_func
        self.learning_rate = learning_rate
        self.layers = []
    def add_layer(self,l):
        self.layers.append(l)
    def learn(self,x,y):
        x_in = x
        for layer in self.layers:
            x_in = layer.forward(x_in)
        final_out = x_in
        self.loss = self.loss_func(final_out,y)
        self.prim_loss = self.prime_loss_func(final_out,y)

        dy = self.prim_loss

        for layer in self.layers[-1::-1]:
            dy = layer.backward(dy)
            layer.apply_gradients(self.learning_rate)
        return self.loss


l = layer(2,3,nop,prime_nop)
l2 = layer(3,1,nop,prime_nop)

n = network(square_loss,prime_square_loss)


n.add_layer(l)
n.add_layer(l2)
x = np.array([[1,1]]).reshape(2,-1)
y = np.array([3]).reshape(1,-1)
x2 = np.array([[2,4]]).reshape(2,-1)
y2 = np.array([10]).reshape(1,-1)

for _ in range(1):
    loss = n.learn(x,y)
    print loss
print l2.weights
print l2.bias
# for _ in range(880):
#     o=l.forward(x)
#     dx = l.backward(o-y)
#     l.apply_gradients(0.01)
#     o=l.forward(x2)
#     dx = l.backward(o-y2)
#     l.apply_gradients(0.01)
# final=l.forward(x2)

# print(final)
# print(l.weights)
# print(l.bias)


