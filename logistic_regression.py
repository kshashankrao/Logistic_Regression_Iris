import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
LR = 0.001
epochs = 800
display = 100
iris = pd.read_csv('Iris.csv')
iris = iris.drop('Id', axis=1)
iris['Species'] = iris['Species'].replace(['Iris-setosa', 'Iris-versicolor','Iris-virginica'], [1, 2, 3])
iris.head(5)
X = iris.loc[:, ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
y = iris.loc[:, ['Species']]
oneHot = OneHotEncoder()
oneHot.fit(X)
X = oneHot.transform(X).toarray()
oneHot.fit(y)
y = oneHot.transform(y).toarray()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state=0)
print("Shape of X_train: ", X_train.shape)
print("Shape of predtrain: ", y_train.shape)
print("Shape of X_test: ", X_test.shape)
print("Shape of predtest", y_test.shape)
X = tf.placeholder(tf.float32, [None, 15])
y = tf.placeholder(tf.float32, [None, 3])
W = tf.Variable(tf.zeros([15, 3]))
b = tf.Variable(tf.zeros([3]))
pred = tf.nn.softmax(tf.add(tf.matmul(X, W), b))
with tf.name_scope("cost"):
    cost = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=pred)
with tf.name_scope("Gradient_Descent"):
    #optimizer = tf.train.GradientDescentOptimizer(LR).minimize(cost)
    optimizer = tf.train.AdamOptimizer(LR).minimize(cost)
with tf.name_scope("Training_session"):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        avg_cost = 0
        for epoch in range(epochs):
            cost_in_each_epoch = 0
            _, c = sess.run([optimizer, cost], feed_dict={X: X_train, y: y_train})
            avg_cost += c / epochs
            if (epoch+1) % display == 0:
                print("Epoch: {}".format(epoch + 1), "cost={}".format(avg_cost))
        print("Optimization Finished!")
        # Test model
        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        # Calculate accuracy for 3000 examples
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        print("Accuracy:", accuracy.eval({X: X_test, y: y_test}))
