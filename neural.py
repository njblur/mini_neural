import numpy as np

def nop(x):
    return x
def prime_nop(x):
    return np.ones(x.shape)
def sigmoid(x):
    x[x<-20] = -20
    x[x>20] = 20
    return 1.0/(1.0+np.exp(-x))
def prime_sigmoid(a):
    return a*(1-a)
def tanh(x):
    x[x<-20] = -20
    x[x>20] = 20
    return (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
def prime_tanh(a):
    return 1-a**2
def softmax(x):
    maxx = np.max(x,axis=0,keepdims=0)
    regx = x-maxx
    exp = np.exp(regx)
    sum = np.sum(exp,axis=0,keepdims=0)
    y = exp/sum
    return y
def prime_softmax(a):
    return a*(1-a) #softmax prime be calculated by loss prime directly
class activation:
    def __init__(self,active,prime_active):
        self.active = active
        self.prime_active = prime_active

activations = dict()
activations["sigmoid"] = activation(sigmoid,prime_sigmoid)
activations["linear"] = activation(nop,prime_nop)
activations["softmax"] = activation(softmax,prime_softmax)
activations["tanh"] = activation(tanh,prime_tanh)
class layer:
    def __init__(self,input_size,output_size,active,bias_rate=1.0,weight_decay=0.0001):
        a = activations[active]
        self.fn_activate = a.active
        self.fn_prime_activate = a.prime_active
        self.weights = np.random.randn(output_size,input_size)/10.0
        self.d_weights = np.zeros((output_size,input_size))
        self.bias = np.random.randn(output_size,1)
        self.d_bias = np.zeros((output_size,1))
        self.m_weights = np.zeros_like(self.weights)
        self.m_bias = np.zeros_like(self.bias)
        self.bias_rate = bias_rate
        self.weight_decay = weight_decay
    def forward(self,x):
        self.x = x
        self.z = np.matmul(self.weights,x)+self.bias*self.bias_rate
        self.out = self.fn_activate(self.z)
        return self.out
    def backward(self,y):
        self.batch_size = len(y[0])
        if(self.fn_activate == softmax):
            self.dy = y
        else:
            self.dy = self.fn_prime_activate(self.out) * y
        self.d_weights = np.matmul(self.dy,self.x.T)
        self.d_bias = np.sum(self.dy,axis=1,keepdims=1)
        self.dx = np.matmul(self.weights.T,self.dy)

        return self.dx
    def apply_gradients(self,learning_rate):
        self.m_weights = self.m_weights*0.9 + self.d_weights*0.1
        self.m_bias = self.m_bias*0.9 + self.d_bias*0.1
        self.weights = self.weights - self.m_weights*learning_rate - self.weights*self.weight_decay
        self.bias -= self.m_bias*learning_rate
        # self.weights = self.weights - self.d_weights*learning_rate - self.weights*self.weight_decay
        # self.bias -= self.d_bias*learning_rate

def square_loss(a,y):
    l = np.sum((a-y)*(a-y))/2
    return np.prod(l)
def prime_square_loss(a,y):
    return (a-y)

def sigmoid_loss(a,y):
    a = a.clip(1e-9,1-1e-9)
    l = -y*np.log(a)-(1-y)*np.log(1-a)
    return np.sum(l)
def prime_sigmoid_loss(a,y):
    a = a.clip(1e-9,1.0-1e-9)
    return (1-y)/(1-a)-y/a

def softmax_loss(a,y):
    a = a.clip(1e-9,1.0)
    l = -y*np.log(a)
    return np.sum(l)
def prime_softmax_loss(a,y):
    return a-y
class loss_func:
    def __init__(self,loss,prime_loss):
        self.loss = loss
        self.prime_loss = prime_loss
loss_funcs = dict()
loss_funcs["square"] = loss_func(square_loss,prime_square_loss)
loss_funcs["sigmoid"] = loss_func(sigmoid_loss,prime_sigmoid_loss)
loss_funcs["softmax"] = loss_func(softmax_loss,prime_softmax_loss)
class network:
    def __init__(self,loss,learning_rate=0.0001):
        f_loss = loss_funcs[loss]
        self.loss_func = f_loss.loss
        self.prime_loss_func = f_loss.prime_loss
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
    asoft = np.array([[0.1],[0.9],[0.01]])
    ysoft = np.array([[0],[1],[0]])
    loss = softmax_loss(asoft,ysoft)
    ploss = prime_softmax_loss(asoft,ysoft)

    print loss
    print ploss
    sm = sigmoid_loss(a,1)
    psm = prime_sigmoid_loss(a,1)
    print y
    print py

    print sm
    print psm

    z = np.array([135,138,40]).T
    maxz = softmax(z)
    print maxz