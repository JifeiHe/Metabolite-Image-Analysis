import tensorflow as tf
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

dir = "/"
os.chdir(dir)

#import Balanced data
bbs_non_label = np.loadtxt('train2928.txt')
label = np.loadtxt('label2928.txt')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0, 10), copy=True)
scaler.fit(bbs_non_label)

bbs_non_label_s = scaler.transform(bbs_non_label)

x_train, x_test, y_train, y_test = train_test_split(bbs_non_label_s, label, test_size=0.3, random_state=0)

# input
x = tf.placeholder("float", [None, 800])
y_ = tf.placeholder("float", [None, 2])
y_list=tf.placeholder("float", [None, 1])

# inference
W = tf.Variable(tf.zeros([800, 2]))

b = tf.Variable(tf.zeros([1, 2]))
matm=tf.matmul(x,W)

y = tf.nn.softmax(tf.matmul(x,W) + b)
#y_list = tf.argmax(y,axis=1)


# loss
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_,logits= y)

# training
train_step = tf.train.GradientDescentOptimizer(1e-3).minimize(cross_entropy)


# training cycles
sess = tf.Session()
sess.run(tf.global_variables_initializer())

batch_size = 1000

for i in range(10):
    random_select = np.random.randint(0,len(y_train), batch_size)
    xs = [x_train[k] for k in random_select]
    ys = [y_train[k] for k in random_select]

    batch_xs = np.array(xs)
    batch_ys = np.array(ys)

    for j in range(800):
        sess.run((train_step, y), feed_dict={x: batch_xs, y_: batch_ys})
        #sess.run(y_list,feed_dict={x: x_test, y_: y_test})
        #_, y_pred = sess.run((train_step, y), feed_dict={x: batch_xs, y_: batch_ys})
    correct_prediction = tf.equal(tf.argmax(y,axis=1), tf.argmax(y_,axis=1)) # y=(m*c) so axis=1 along c
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print "batch%d: "%i, sess.run(accuracy, feed_dict={x: x_test, y_: y_test})
    #print sess.run(correct_prediction, feed_dict={x: batch_xs, y_: batch_ys})
    #print sess.run(accuracy, feed_dict={x: batch_xs, y_: batch_ys})

    print  sess.run(y, feed_dict={x: x_test, y_: y_test})

print sess.run(accuracy, feed_dict={x: x_test, y_: y_test})

img_shape = [40, 20]
def plot_weights():
    # Get the values for the weights from the TensorFlow variable.
    weights = sess.run(W)

    # Get the lowest and highest values for the weights.
    # This is used to correct the colour intensity across
    # the images so they can be compared with each other.
    w_min = np.min(weights)
    w_max = np.max(weights)

    fig, axes = plt.subplots(1, 2)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        # Only use the weights for the first 10 sub-plots.
        if i<10:
            # Get the weights for the i'th digit and reshape it.
            # Note that w.shape == (img_size_flat, 10)
            #image = W[:, i].reshape(img_shape)
            image = np.reshape(weights[:, i],[40,20])
            print (np.shape(image))
            #image=np.array(40*20, dtype=float)


            # Set the label for the sub-plot.
            ax.set_xlabel("Weights: {0}".format(i))

            # Plot the image.
           # ax.imshow(imread(image), vmin=w_min, vmax=w_max, cmap='seismic')
            cax=ax.imshow(image, vmin=w_min, vmax=w_max, cmap='seismic')

        # Remove ticks from each sub-plot.
#         ax.set_xticks([])
#         ax.set_yticks([])


    # Ensure the plot is shown correctly with multiple plots
#     # in a single Notebook cell.
    cbar=fig.colorbar(cax,ticks=[-1, 0, 1])
    plt.show()
plot_weights()
