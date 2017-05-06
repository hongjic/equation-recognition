from glob import glob
from skimage import io, transform, img_as_float
import sys
import numpy as np
import tensorflow as tf
import shelve
import random


class SymbolData:
    img_labels = shelve.open("img-labels")

    def __init__(self, dir):
        image_paths = glob(dir + "/*.png")
        self.data_size = len(image_paths)
        # read images of symbols and their labels
        self.labels = np.zeros((self.data_size, 40))
        self.images = np.zeros((self.data_size, 64, 64))
        self.names = []
        for i in range(self.data_size):
            path = image_paths[i]
            fname = path.split("/")[-1]
            float_img = img_as_float(io.imread(path, as_grey=True))
            image = self.__normalize(float_img)
            self.images[i] = image
            self.names.append(fname)
            self.labels[i][self.img_labels[fname]] = 1

    def __normalize(self, image):
        rows, cols = image.shape
        padding = image
        if rows > cols:
            left = (rows - cols) / 2
            right = rows - cols - left
            padding = np.lib.pad(image, ((0, 0), (left, right)), "constant", constant_values=(0, 0))
        elif cols > rows:
            top = (cols - rows) / 2
            bottom = cols - rows - top
            padding = np.lib.pad(image, ((top, bottom), (0, 0)), "constant", constant_values=(0, 0))
        return transform.resize(padding, (64, 64))

    def get_batch(self, batch_size):
        sample = random.sample(range(self.data_size), batch_size)
        batch_images = np.zeros((batch_size, 64, 64))
        batch_labels = np.zeros((batch_size, 40))
        for i in range(batch_size):
            batch_images[i] = self.images[sample[i]]
            batch_labels[i] = self.labels[sample[i]]
        return batch_images, batch_labels

    def get_all_images(self):
        labels = []
        for name in self.names:
            labels.append(self.img_labels[name])
        return self.images, self.names, labels


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    # strides = 1, boundary padding = "SAME"
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding="SAME")

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME")

# CNN
x = tf.placeholder(tf.float32, [None, 64, 64])
x_image = tf.reshape(x, [-1, 64, 64, 1])
y_ = tf.placeholder(tf.float32, [None, 40])

# convolutional layer 1
w_conv1 = weight_variable([5, 5, 1, 8])
b_conv1 = bias_variable([8])
h_conv1 = tf.nn.relu(conv2d(x_image, w_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# convolutional layer 2
w_conv2 = weight_variable([5, 5, 8, 16])
b_conv2 = bias_variable([16])
h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# convolutional layer 3
w_conv3 = weight_variable([5, 5, 16, 32])
b_conv3 = bias_variable([32])
h_conv3 = tf.nn.relu(conv2d(h_pool2, w_conv3) + b_conv3)
h_pool3 = max_pool_2x2(h_conv3)

#convolutional layer 4
w_conv4 = weight_variable([5, 5, 32, 64])
b_conv4 = bias_variable([64])
h_conv4 = tf.nn.relu(conv2d(h_pool3, w_conv4) + b_conv4)

h_conv4_flat = tf.reshape(h_conv4, [-1, 8 * 8 * 64])

# densely connected layer, 1024 nuerals 
w_fc1 = weight_variable([8 * 8 * 64, 1024])
b_fc1 = bias_variable([1024])
h_fc1 = tf.nn.relu(tf.matmul(h_conv4_flat, w_fc1) + b_fc1)

# drop out
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# readout layer
w_fc2 = weight_variable([1024, 40])
b_fc2 = bias_variable([40])
y_conv = tf.matmul(h_fc1_drop, w_fc2) + b_fc2


def train(_):
    train_path = _[1]
    symbol_data = SymbolData(train_path)
    # training config
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
    saver = tf.train.Saver()
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    # training starts
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    for i in range(3000):
        batch_images, batch_labels = symbol_data.get_batch(30)
        if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={x: batch_images, y_: batch_labels, keep_prob: 1.0})
            print("step %d, training accuracy %g"%(i, train_accuracy))
        train_step.run(feed_dict={x: batch_images, y_: batch_labels, keep_prob: 0.5})
    model_path = saver.save(sess, "model/model.ckpt")
    print("Model saved at: " + model_path)


def test(_):
    # test
    test_path = _[1]
    symbol_data = SymbolData(test_path)
    sess = tf.InteractiveSession()
    saver = tf.train.Saver()
    saver.restore(sess, "model2/model.ckpt")
    print("Model restored.")
    test_images, image_names, labels = symbol_data.get_all_images()
    size = len(test_images)
    categories = tf.argmax(y_conv, 1).eval(feed_dict={x: test_images, keep_prob: 1.0})
    output = open("output.txt", "w")
    corrects = 0
    set1 = set([5, 25, 19])
    set2 = set([32, 8])
    for i in range(0, len(categories)):
        output.write(image_names[i] + "\t" + str(categories[i]) + "\t" + str(labels[i]) + "\n")
        if categories[i] == labels[i]:
            corrects += 1
        else:
            if categories[i] in set1 and labels[i] in set1:
                corrects += 1
            if categories[i] in set2 and labels[i] in set2:
                corrects += 1
    print("Symbol Recognition Accuracy: %d / %d = %g"%(corrects, size, float(corrects) / size))
    output.close()


if __name__ == "__main__":
    if len(sys.argv) <= 2:
        print("options: \n\t-train <image path>\n\t-test <image path>")
    else:
        option = sys.argv[1]
        path = sys.argv[2]
        if option == "-train":
            tf.app.run(main=train, argv=[sys.argv[0], path])
        else:
            tf.app.run(main=test, argv=[sys.argv[0], path])
