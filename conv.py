import numpy as np 
import IPython
class conv2d:
    def __init__(self,filter,stride,padding):
        self.filter = filter
        self.filter_linear = self.filter.reshape(-1,self.filter.shape[-1])
        self.bias = np.zeros(shape=[filter.shape[-1]],dtype=float)
        self.stride = stride
        self.padding = padding
    def forward(self,x):
        padding = self.padding
        stride = self.stride
        self.x = x
        ih,iw,ic = x.shape
        h,w,c = ih+2*padding,iw+2*padding,ic
        self.expand = np.zeros((h,w,c),dtype='float')
        self.expand[padding:-padding,padding:-padding,:] = x
        fh,fw,fic,foc = self.filter.shape
        assert(c == fic)
        target_h = (h-fh)/stride+1
        target_w = (w-fw)/stride+1
        garden = np.zeros(shape=[target_h,target_w,fh,fw,c])
        for i in range(target_h):
            for j in range(target_w):
                garden[i,j,:,:,:] = self.expand[i*stride:i*stride+fh,j*stride:j*stride+fw,:]

        self.img2col = garden.reshape(target_h*target_w,fh*fw*c)
        y = self.img2col.dot(self.filter_linear)+self.bias
        return y.reshape(target_h,target_w,foc)
    def backward(self,dy):
        dy_col = dy.reshape(-1,dy.shape[-1])
        self.dbias = np.sum(dy_col,axis=0)
        dfilter_linear = self.img2col.T.dot(dy_col)
        self.dfilter = dfilter_linear.reshape(self.filter.shape)
        dimg2col = dy_col.dot(self.filter_linear.T)
        fh,fw,fci,fco = self.filter.shape
        th,tw,tc = dy.shape
        dgarden = dimg2col.reshape(th,tw,fh,fw,fci)
        dexpand = np.zeros_like(self.expand)
        stride = self.stride
        padding = self.padding
        for i in range(th):
            for j in range(tw):
                dexpand[i*stride:i*stride+fh,j*stride:j*stride+fw,:] += dgarden[i,j,:,:,:]
        return dexpand[padding:-padding,padding:-padding,:]
    def apply_gradients(self,learning_rate):
        self.filter -= self.dfilter*learning_rate
        self.bias -= self.dbias*learning_rate*0.0001

if __name__ == '__main__':
    image = np.ones(shape=[48,64,3])
    filter = np.ones(shape=[5,5,3,7])
    stride = 1
    padding = filter.shape[0]/2
    conv = conv2d(filter,stride,padding)
    out=conv.forward(image)
    dout = np.random.random(size=out.shape)
    IPython.embed()
