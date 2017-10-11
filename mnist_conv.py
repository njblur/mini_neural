
import numpy as np
import IPython
import mnist_reader
import neural
import conv
batch_size = 20
label_num = 10
data = mnist_reader.load_mnist(train_dir='data',reshape=False)
train = data.train
test = data.test

size = train.num_examples
idx = range(batch_size)


filter0 = np.random.standard_normal(size=[3,3,1,16])*0.1
filter = np.random.standard_normal(size=[3,3,16,16])*0.1
conv_layer0 = conv.conv2d(filter0,2,1)
relu0 = conv.relu()
dropout0 = conv.dropout(0.5)
conv_layer = conv.conv2d(filter,2,1)
relu = conv.relu()

hidden = 128
l1 = neural.layer(28*28/4/4*16,hidden,"relu",weight_decay=0.0)
l2 = neural.layer(hidden,label_num,"softmax",weight_decay=0.0)

centers = np.random.standard_normal(size=[hidden,label_num])*0.001

loop = size//batch_size

epoch = 10
learning_rate = 0.001
for k in range(epoch):
    for j in range(loop):
        images,labels = train.next_batch(batch_size)
        ori_labels = labels
        l = np.zeros([label_num,batch_size])
        l[[labels],idx] = 1
        data = images
        labels = l
        conv_out0 = conv_layer0.forward(data)
        reludata0 = relu0.forward(conv_out0)
        conv_out = conv_layer.forward(reludata0)
        reludata = relu.forward(conv_out)
        linear = reludata.reshape(batch_size,-1).transpose(1,0) # my bad, our basic neural cell need Dxbatch input
        l1_out = l1.forward(linear)
        dropoutdata0 = dropout0.forward(l1_out)
        l2_out = l2.forward(dropoutdata0)
        loss = neural.softmax_loss(l2_out,labels).sum()/batch_size
        print 'loss is ' + str(loss) + '  '+str(j*100.0/loop)+'%'
        ## calculate center loss
        batch_centers = centers[:,ori_labels]
        center_diff = l1_out-batch_centers
        center_loss = np.square(center_diff)/2/batch_size
        d_center_loss = l1_out - batch_centers
        d_l1_center = center_diff
        d_centers = -center_diff
        total_center_loss = np.sum(np.sum(center_loss,axis=0))
        print "center loss is %f"%total_center_loss
        ## end calculate center loss

        dloss = neural.prime_softmax_loss(l2_out,labels)
        dl2_out = l2.backward(dloss)
        ddropout0 = dropout0.backward(dl2_out)
        ## apply center loss
        if k > 5:
            ddropout0 += d_l1_center*0.01
            for i in range(batch_size):
                center_idx = ori_labels[i]
                centers[:,center_idx] -= d_center_loss[:,i]*learning_rate*0.01
        dlinear = l1.backward(ddropout0)
        dlinear = dlinear.T
        dreluout = dlinear.reshape(reludata.shape)
        dconv_out = relu.backward(dreluout)
        ddata = conv_layer.backward(dconv_out)
        drelu0 = relu0.backward(ddata)
        dconv_out0 = conv_layer0.backward(drelu0)
        conv_layer.apply_gradients(learning_rate)
        conv_layer0.apply_gradients(learning_rate)
        l1.apply_gradients(learning_rate)
    if (k > 0):
        learning_rate =0.002

    correct = 0
    test_size = test.num_examples
    for i in range(test_size//batch_size):
        images,labels = test.next_batch(batch_size)
        data = images
        label = labels
        conv_out0 = conv_layer0.forward(data)
        reludata0 = relu0.forward(conv_out0)
        conv_out = conv_layer.forward(reludata0)
        reludata = relu.forward(conv_out)
        linear = conv_out.reshape(batch_size,-1).T
        l1_out = l1.forward(linear)
        l2_out = l2.forward(l1_out)
        n = np.argmax(l2_out,axis=0)
        correct += np.count_nonzero(n == label )
    # IPython.embed()
    print "{} correct, rate is {}".format(correct,correct*1.0/test_size)
