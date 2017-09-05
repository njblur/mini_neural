
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


filter0 = np.random.standard_normal(size=[3,3,1,5])*0.1
filter = np.random.standard_normal(size=[3,3,5,5])*0.1
conv_layer0 = conv.conv2d(filter0,2,1)
relu0 = conv.relu()
conv_layer = conv.conv2d(filter,2,1)
relu = conv.relu()

l1 = neural.layer(28*28/4/4*5,10,"sigmoid",weight_decay=0.001)

loop = size//batch_size

epoch = 3
learning_rate = 0.02
for _ in range(epoch):
    for _ in range(loop):
        images,labels = train.next_batch(batch_size)
        l = np.zeros([batch_size,label_num])
        l[idx,[labels]] = 1
        for i in range(batch_size):
            data = images[i]
            label = l[i].reshape(-1,1)
            conv_out0 = conv_layer0.forward(data)
            reludata0 = relu0.forward(conv_out0)
            conv_out = conv_layer.forward(reludata0)
            reludata = relu.forward(conv_out)
            linear = reludata.reshape(-1,1)
            l1_out = l1.forward(linear)
            loss = neural.sigmoid_loss(l1_out,label)
            print 'loss is ' + str(loss)
            dloss = neural.prime_sigmoid_loss(l1_out,label)
            dlinear = l1.backward(dloss)
            dreluout = dlinear.reshape(reludata.shape)
            dconv_out = relu.backward(dreluout)
            ddata = conv_layer.backward(dconv_out)
            drelu0 = relu0.backward(ddata)
            dconv_out0 = conv_layer0.backward(drelu0)
            conv_layer.apply_gradients(learning_rate)
            conv_layer0.apply_gradients(learning_rate)
            l1.apply_gradients(learning_rate)
    learning_rate *= 0.3
         

    correct = 0
    test_size = test.num_examples
    for i in range(test_size):
            images,labels = test.next_batch(1)
            data = images[0]
            label = labels[0]
            conv_out0 = conv_layer0.forward(data)
            reludata0 = relu0.forward(conv_out0)
            conv_out = conv_layer.forward(reludata0)
            reludata = relu.forward(conv_out)
            linear = conv_out.reshape(-1,1)
            l1_out = l1.forward(linear)
            # l2_out = l2.forward(l1_out)
            n = np.argmax(l1_out,axis=0)
            if(n[0] == label ):
                correct += 1
    print "{} correct, rate is {}".format(correct,correct*1.0/test_size)
    # IPython.embed()