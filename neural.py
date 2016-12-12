import numpy as np

def nop(x):
    return x
def prime_nop(x):
    return np.ones(x.shape)
def sigmoid(x):
    return 1/(1+np.exp(-x))
def prime_sigmoid(x):
    return sigmoid(x)*(1-sigmoid(x))
def softmax(x):
    exp = np.exp(x)
    sum = np.sum(exp)
    y = exp/sum
    return y
def prime_softmax(x):
    return softmax(x)*(1-softmax(x))
class activation:
    def __init__(self,active,prime_active):
        self.active = active
        self.prime_active = prime_active

activations = dict()
activations["sigmoid"] = activation(sigmoid,prime_sigmoid)
activations["liner"] = activation(nop,prime_nop)
activations["softmax"] = activation(softmax,prime_softmax)

class layer:
    def __init__(self,input_size,output_size,active,bias_rate=1.0):
        a = activations[active]
        self.fn_activate = a.active
        self.fn_prime_activate = a.prime_active
        self.weights = np.random.randn(output_size,input_size)
        self.d_weights = np.zeros((output_size,input_size))
        self.bias = np.random.randn(output_size,1)
        self.d_bias = np.zeros((output_size,1))
        self.bias_rate = bias_rate
    def forward(self,x):
        self.x = x
        self.z = np.matmul(self.weights,x)+self.bias*self.bias_rate
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

def sigmoid_loss(a,y):
    return -y*np.log(a)-(1-y)*np.log(1-a)
def prime_sigmoid_loss(a,y):
    return (1-y)/(1-a)-y/a

class network:
    def __init__(self,loss_func,prime_loss_func,learning_rate=0.0001):
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
    def eval(self,x):
        x_in = x
        for layer in self.layers:
            x_in = layer.forward(x_in)
        return x_in

    def set_learning_rate(self,learning_rate):
        self.learning_rate = learning_rate

if __name__ == "__main__":
    print "test start from here"
    x = np.array([[1],[5],[7]])
    y = softmax(x)
    py = prime_softmax(x)

    a = 0.8
    sm = sigmoid_loss(a,1)
    psm = prime_sigmoid_loss(a,1)
    print y
    print py

    print sm
    print psm