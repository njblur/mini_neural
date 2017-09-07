import numpy as np 
import IPython
class relu:
    def forward(self,x):
        self.mask = (x < 0) 
        x[self.mask] = 0
        return x

    def backward(self,dy):
        dy[self.mask] = 0
        return dy

    def apply_gradients(self,learning_rate):
        return
class dropout:
    def __init__(self,keep):
        self.keep = keep
    def forward(self,x):
        linear = x.reshape(-1)
        self.outs = np.random.choice(len(linear),size=int(len(linear)*(1-self.keep))) 
        linear[self.outs] = 0
        return x

    def backward(self,dy):
        linear = dy.reshape(-1)
        linear[self.outs] = 0
        return dy

    def apply_gradients(self,learning_rate):
        return
class conv2d:
    def __init__(self,filter,stride,padding):
        self.filter = filter
        self.filter_linear = self.filter.reshape(-1,self.filter.shape[-1])
        self.bias = np.zeros(shape=[filter.shape[-1]],dtype=float)
        self.stride = stride
        self.padding = padding
        self.m_filter = np.zeros_like(self.filter)
        self.m_bias = np.zeros_like(self.bias)
    def forward(self,x):
        padding = self.padding
        stride = self.stride
        self.x = x
        b,ih,iw,ic = x.shape
        h,w,c = ih+2*padding,iw+2*padding,ic
        self.expand = np.zeros((b,h,w,c),dtype='float')
        self.expand[:,padding:-padding,padding:-padding,:] = x
        fh,fw,fic,foc = self.filter.shape
        assert(c == fic)
        target_h = (h-fh)/stride+1
        target_w = (w-fw)/stride+1
        garden = np.zeros(shape=[b,target_h,target_w,fh,fw,c])
        for i in range(target_h):
            for j in range(target_w):
                garden[:,i,j,:,:,:] = self.expand[:,i*stride:i*stride+fh,j*stride:j*stride+fw,:]

        self.img2col = garden.reshape(b,target_h*target_w,fh*fw*c)
        y = self.img2col.dot(self.filter_linear)+self.bias
        return y.reshape(b,target_h,target_w,foc)
    def backward(self,dy):
        b,th,tw,tc = dy.shape
        dy_col = dy.reshape(b,-1,dy.shape[-1])
        self.dbias = np.mean(np.sum(dy_col,axis=0),axis=0)
        dfilter_linear = np.matmul(self.img2col.transpose(0,2,1),dy_col)
        dfilter_linear = np.mean(dfilter_linear,axis=0)
        self.dfilter = dfilter_linear.reshape(self.filter.shape)
        dimg2col = np.matmul(dy_col,self.filter_linear.T)
        fh,fw,fci,fco = self.filter.shape

        dgarden = dimg2col.reshape(b,th,tw,fh,fw,fci)
        dexpand = np.zeros_like(self.expand)
        stride = self.stride
        padding = self.padding
        for i in range(th):
            for j in range(tw):
                dexpand[:,i*stride:i*stride+fh,j*stride:j*stride+fw,:] += dgarden[:,i,j,:,:,:]
        return dexpand[:,padding:-padding,padding:-padding,:]
    def apply_gradients(self,learning_rate):
        self.m_bias = self.m_bias*0.9 + self.dbias*0.1
        self.m_filter = self.m_filter*0.9 + self.dfilter*0.1
        self.filter -= self.m_filter*learning_rate
        self.bias -= self.m_bias*learning_rate
        # self.filter -= self.dfilter*learning_rate
        # self.bias -= self.dbias*learning_rate
if __name__ == '__main__':
    image = np.ones(shape=[48,64,3])
    filter = np.ones(shape=[5,5,3,7])
    stride = 1
    padding = filter.shape[0]/2
    conv = conv2d(filter,stride,padding)
    out=conv.forward(image)
    dout = np.random.random(size=out.shape)
    IPython.embed()
