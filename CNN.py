import tensorflow as tf
import numpy as np
import os
import pandas as pd
from numpy import array
from numpy import argmax
from keras.utils import to_categorical
import matplotlib.pyplot as plt
%matplotlib inline

#Loading data
dir = "/"
os.chdir(dir)

Img1 = np.loadtxt('Results-NP_C1_N/Images.txt')
Img2 = np.loadtxt('Results-NP_C2_N/Images.txt')
Img3 = np.loadtxt('Results-NP_C3_N/Images.txt')
Img4 = np.loadtxt('Results-NP_C4_N/Images.txt')
print Img3.shape

Label1 = np.loadtxt('Results-NP_C1_N/Final-List.txt')
Label2 = np.loadtxt('Results-NP_C2_N/Final-List.txt')
Label3 = np.loadtxt('Results-NP_C3_N/Final-List.txt')
Label4 = np.loadtxt('Results-NP_C4_N/Final-List.txt')
print Label1.shape

ImageN=299 #Each sample has 299 images
SampleN=4
Img_list = [Img1, Img2, Img3, Img4]
Img = []
for i in Img_list:
    for j in range(ImageN):
        Img += [i[j]]
X = np.array(Img)

Label_list = [Label1, Label2, Label3, Label4]
Label = []
for i in Label_list:
    for j in range(ImageN):
        Label += [i[j]]

#one hot encoded y
y = to_categorical(np.array(Label)[:,2])
print X.shape, y.shape

plt.imshow(X[1].reshape((60, 12)), cmap = 'jet')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0, 10), copy=True)
scaler.fit(X)

scaler_X = scaler.transform(X)

x_train, x_test, y_train, y_test = train_test_split(scaler_X, y, test_size=0.3, random_state=0)

x = tf.placeholder("float", [None, 720])
y_ = tf.placeholder("float", [None, 4])

x_image = tf.reshape(x, [-1,60,12,1])
print "x_image=", x_image

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

W_conv1 = weight_variable([5, 5, 1, 64])
b_conv1 = bias_variable([64])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, 64, 128])
b_conv2 = bias_variable([128])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weight_variable([15 * 3 * 128, 250])
b_fc1 = bias_variable([250])

h_pool2_flat = tf.reshape(h_pool2, [-1, 15*3*128])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([250, 4])
b_fc2 = bias_variable([4])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
y_conv2 = tf.argmax(y_conv,1)

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_,logits= y_conv ))

train_step = tf.train.AdamOptimizer(2e-5).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# for img in x_train:
#     img *= 255.0/img.max()

# for img in x_test:
#     img *= 255.0/img.max()


#batch_size = 4000
error = []
_result = []


epoch = 3000
batch_size = 100

for j in range(epoch):
    random_select = np.random.randint(0,len(y_train), batch_size)
    xs = [x_train[k] for k in random_select]
    ys = [y_train[k] for k in random_select]

    batch_xs = np.array(xs)
    batch_ys = np.array(ys)

    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 0.8})
    train_accuracy, loss, y_soft = sess.run([accuracy,cross_entropy, y_conv2]
                                            , feed_dict={x:batch_xs, y_: batch_ys, keep_prob: 1.0})


    _result.append(y_soft)
    error.append(loss)

    if j%100 == 0:
        print("step %d, training accuracy %g"%(j, train_accuracy))
        print("loss : ", loss)
        print("test accuracy %g"% sess.run(accuracy, feed_dict={x: x_test, y_: y_test, keep_prob: 1.0}))
        #print( sess.run(W_fc2, feed_dict={x: x_test, y_: y_test, keep_prob: 1.0}))


# train_accuracy, loss, y_soft, tf = sess.run([accuracy,cross_entropy, y_conv2, correct_prediction]
#                                             , feed_dict={x: x_test, y_: y_test, keep_prob: 1})
