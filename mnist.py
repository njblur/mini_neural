import mnist_reader
import numpy as np
import neural
batch_size = 20
label_num = 10
data = mnist_reader.load_mnist(train_dir='data')
train = data.train
test = data.test

size = train.num_examples
idx = range(batch_size)
print size


l1 = neural.layer(784,80,"sigmoid",weight_decay=0.001)
l2 = neural.layer(80,10,"sigmoid",weight_decay=0.001)

net = neural.network("sigmoid",learning_rate=0.9)
net.add_layer(l1)
net.add_layer(l2)
loop = size//batch_size
epoch = 5
for _ in range(epoch):
    for _ in range(loop):
        images,labels = train.next_batch(batch_size)
        l = np.zeros([label_num,batch_size])
        images = images.T
        l[[labels],idx] = 1
        loss = net.learn(images,l)
    net.set_learning_rate(net.learning_rate*0.8)

    # print loss

correct = 0
test_size = test.num_examples
for i in range(test_size):
    images,labels = test.next_batch(1)
    images = images.T
    o = net.eval(images)
    n = np.argmax(o,axis=0)
    if(i%100 == 0):
        print n
        print labels
    if(n[0] == labels[0] ):
        correct += 1

print "{} correct, rate is {}".format(correct,correct*1.0/test_size)