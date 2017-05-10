from glob import glob
from skimage import io, transform, img_as_float
import sys
import numpy as np
import tensorflow as tf
import imageutil as util
import shelve
import random


class SymbolData:


    def __init__(self, dir, custom=False):
        self.img_labels = shelve.open("img-labels")
        image_paths = glob(dir + "/*.png")
        self.data_size = len(image_paths)
        # read images of symbols and their labels
        if not custom:
            self.labels = np.zeros((self.data_size, 40))
        self.images = np.zeros((self.data_size, 64, 64))
        self.names = []
        for i in range(self.data_size):
            path = image_paths[i]
            fname = path.split("/")[-1]
            float_img = img_as_float(io.imread(path, as_grey=True))
            image = util.normalize(float_img)
            self.images[i] = image
            self.names.append(fname)
            if not custom:
                self.labels[i][self.img_labels[fname]] = 1


    def get_training_batch(self, batch_size):
        sample = random.sample(range(self.data_size), batch_size)
        batch_images = np.zeros((batch_size, 64, 64))
        batch_labels = np.zeros((batch_size, 40))
        for i in range(batch_size):
            batch_images[i] = self.images[sample[i]]
            batch_labels[i] = self.labels[sample[i]]
        return batch_images, batch_labels

    def get_all_images(self, custom=False):
        if custom:
            return self.images, self.names
        labels = []
        for name in self.names:
            labels.append(self.img_labels[name])
        return self.images, self.names, labels


class EquationFCN:

    def __init__(self, equation_test):
        self.equation_test = equation_test
        self.buildCNN()
        self.saver = tf.train.Saver()
        self.sess = tf.InteractiveSession()

    def restore_model(self):
        self.saver.restore(self.sess, "model/model.ckpt")
        print("Model restored.")

    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def conv2d(self, x, W, padding="SAME"):
        # strides = 1, boundary padding = "SAME" for feature learning; "Valid" for output
        return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding=padding)

    def max_pool_2x2(self, x):
        return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME")

    def buildCNN(self):
        # CNN
        if self.equation_test:
            x = tf.placeholder(tf.float32, [1, None, None, 1])
        else:
            x = tf.placeholder(tf.float32, [None, 64, 64, 1])
        y_ = tf.placeholder(tf.float32, [None, 40])

        # convolutional layer 1
        w_conv1 = self.weight_variable([5, 5, 1, 8])
        b_conv1 = self.bias_variable([8])
        h_conv1 = tf.nn.relu(self.conv2d(x, w_conv1) + b_conv1)
        h_pool1 = self.max_pool_2x2(h_conv1)

        # convolutional layer 2
        w_conv2 = self.weight_variable([5, 5, 8, 16])
        b_conv2 = self.bias_variable([16])
        h_conv2 = tf.nn.relu(self.conv2d(h_pool1, w_conv2) + b_conv2)
        h_pool2 = self.max_pool_2x2(h_conv2)

        # convolutional layer 3
        w_conv3 = self.weight_variable([5, 5, 16, 32])
        b_conv3 = self.bias_variable([32])
        h_conv3 = tf.nn.relu(self.conv2d(h_pool2, w_conv3) + b_conv3)
        h_pool3 = self.max_pool_2x2(h_conv3)

        #convolutional layer 4
        w_conv4 = self.weight_variable([5, 5, 32, 64])
        b_conv4 = self.bias_variable([64])
        h_conv4 = tf.nn.relu(self.conv2d(h_pool3, w_conv4) + b_conv4)
        # h_conv4_flat = tf.reshape(h_conv4, [-1, 8 * 8 * 64])

        # densely connected layer, 1024 neurals
        w_fc1 = self.weight_variable([8, 8, 64, 1024])
        b_fc1 = self.bias_variable([1024])
        h_fc1 = tf.nn.relu(self.conv2d(h_conv4, w_fc1, padding="VALID") + b_fc1)

        # drop out
        keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

        # readout layer
        w_fc2 = self.weight_variable([1, 1, 1024, 40])
        b_fc2 = self.bias_variable([40])
        y_conv = self.conv2d(h_fc1_drop, w_fc2, padding="VALID") + b_fc2
        self.x = x
        self.y_ = y_
        self.y_conv = y_conv
        self.keep_prob = keep_prob
        if not self.equation_test:
            self.h_readout = tf.reshape(y_conv, [-1, 40])

    def train(self, _):
        train_path = _[1]
        symbol_data = SymbolData(train_path)
        # training config
        cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=self.y_, logits=self.h_readout))
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
        # training starts
        self.sess.run(tf.global_variables_initializer())
        correct_prediction = tf.equal(tf.argmax(self.h_readout, 1), tf.argmax(self.y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        for i in range(2500):
            batch_images, batch_labels = symbol_data.get_training_batch(30)
            if i % 100 == 0:
                train_accuracy = accuracy.eval(feed_dict={
                    self.x: np.reshape(batch_images, (30, 64, 64, 1)),
                    self.y_: batch_labels,
                    self.keep_prob: 1.0
                })
                print("step %d, training accuracy %g"%(i, train_accuracy))
            train_step.run(feed_dict={
                self.x: np.reshape(batch_images, (30, 64, 64, 1)),
                self.y_: batch_labels,
                self.keep_prob: 0.5
            })
        model_path = self.saver.save(self.sess, "model/model.ckpt")
        print("Model saved at: " + model_path)

    def test_standard(self, _):
        # test
        test_path = _[1]
        symbol_data = SymbolData(test_path)
        self.restore_model()
        test_images, image_names, labels = symbol_data.get_all_images()
        size = len(test_images)
        categories = tf.argmax(self.h_readout, 1).eval(feed_dict={
            self.x: np.reshape(test_images, (size, 64, 64, 1)),
            self.keep_prob: 1.0
        })
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

    def test_custom(self, test_images):
        # test_images = _[1]
        size = len(test_images)
        categories = tf.argmax(self.h_readout, 1).eval(feed_dict={
            self.x: np.reshape(test_images, (size, 64, 64, 1)),
            self.keep_prob: 1.0
            })
        return categories

    def equation_recog(self, _):
        # equation recognition
        equation_path = _[1]
        self.restore_model()
        image = img_as_float(io.imread(equation_path, as_grey=True))
        heat_map = self.y_conv.eval(feed_dict={
            self.x: np.reshape(image, (1, image.shape[0], image.shape[1], 1)),
            self.keep_prob: 1.0
        })

        size, rows, cols, channel = heat_map.shape
        category_map = np.zeros((rows, cols), dtype=int)
        # upsample the heatmap and label all pixels.


def symbols_recognition(images):
    fcn = EquationFCN(False)
    fcn.restore_model()
    return fcn.test_custom(images)


if __name__ == "__main__":
    if len(sys.argv) <= 2:
        print("options: \n\t-train <image path>\n\t-test <image path>")
    else:
        option = sys.argv[1]
        path = sys.argv[2]
        if option == "-train":
            fcn = EquationFCN(False)
            tf.app.run(main=fcn.train, argv=[sys.argv[0], path])
        elif option == "-test":
            fcn = EquationFCN(False)
            tf.app.run(main=fcn.test_standard, argv=[sys.argv[0], path])
        elif option == "-equation":
            fcn = EquationFCN(True)
            tf.app.run(main=fcn.equation_recog, argv=[sys.argv[0], path])