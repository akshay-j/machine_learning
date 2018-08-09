import extract_data
import random
import tensorflow as tf

# prepare input data and output labels for neural network
num_epochs = 101
ROOT_DIR_TRAIN = '/home/akshay_joshi/Documents/deeplearn/datasets/BelgiumRoadSigns_Training/'
ROOT_DIR_TEST = '/home/akshay_joshi/Documents/deeplearn/datasets/BelgiumRoadSigns_Testing/'

# obtain sign images and their corresponding labels
labels, images = extract_data.load_data(ROOT_DIR_TRAIN)

# obtaining grayscale, resized images
images = extract_data.prepare_image_for_network(images)

# create tensorflow networks
image_inp = tf.placeholder(dtype = tf.float32, shape = [None, 28, 28], name = "image_inp")
label_oup = tf.placeholder(dtype = tf.int32, shape = [None], name = "label_oup")

# flatten matrix from 28 X 28 to 1 X 784
images_flat = tf.contrib.layers.flatten(image_inp)

# fully connected layers
train_oup = tf.contrib.layers.fully_connected(inputs = images_flat, num_outputs = 62, activation_fn = tf.nn.tanh)

# define a loss function
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = label_oup, logits = train_oup))

# define at optimizer
train_opt = tf.train.AdamOptimizer(learning_rate = 0.001).minimize(loss)

# convert logits to label indices
correct_pred = tf.argmax(train_oup, 1)

# define an accuracy metric
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

sess = tf.Session()

sess.run(tf.global_variables_initializer())

for i in range(num_epochs):
	loss_val, _, accuracy_val = sess.run([loss, train_opt, accuracy], feed_dict={image_inp: images, label_oup: labels})


# testing our data with test dataset
test_labels, test_images = extract_data.load_data(ROOT_DIR_TEST)
test_images = extract_data.prepare_image_for_network(test_images)

# run predictions against full test set
predicted = sess.run([correct_pred], feed_dict={image_inp: test_images})[0]

# calculate correct matches
match_count = sum([int(y==y_) for y, y_ in zip(test_labels, predicted)])

# accuracy calculation
test_accuracy = float(match_count) / len(test_labels)

# print accuracy
print ('Match count:' + str(match_count))
print ("Test accuracy:" + str(test_accuracy))

sess.close()
