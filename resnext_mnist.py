#ResNext Architecture using MNIST dataset
import tensorflow as tf
import numpy as np

batch_size = 100
training_rate = 1e-1
batches = 5000

path = "./graphs/resnext"

def get_data():
    mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    train_data = mnist.train.images
    train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
    eval_data = mnist.test.images
    eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

    # 28 * 28 images, grayscale, already normalized
    return train_data, train_labels, eval_data, eval_labels

def batch_norm(input, depth):
    epsilon = 1e-6
    mean, var = tf.nn.moments(input, axes=[0, 1, 2])
    scale = tf.Variable(tf.ones([depth]))
    beta = tf.Variable(tf.zeros([depth]))
    return tf.nn.batch_normalization(input, mean, var, beta, scale, epsilon)

def grouped_convolution(input, cardinality, filters, strides=(1,1)):
    group = []
    for i in range(cardinality):
        with tf.name_scope("cardinality"+str(i)):
            conv = tf.layers.conv2d(input, filters=filters, kernel_size=(3,3), strides=strides, padding='SAME')
            conv = batch_norm(conv, filters)
            conv = tf.nn.relu(conv)
            group.append(conv)
    return group

#combine residual block "gradient super highway" with inception modules
def res_block(name, input, strides, filters, cardinality, pooling1_1):
    with tf.name_scope(name):
        with tf.name_scope("layer1"):
            #condense depth using 1 * 1 convolution
            conv1 = tf.layers.conv2d(input, filters=pooling1_1, kernel_size=(1,1), strides=(1,1), padding='SAME')
            conv1 = batch_norm(conv1, pooling1_1)
            conv1 = tf.nn.relu(conv1)
        with tf.name_scope("layer2"):
            concat = grouped_convolution(conv1, cardinality=cardinality, filters=pooling1_1, strides=strides)
            concat = tf.concat(concat, axis=3)
        with tf.name_scope("layer3"):
            #inflate depth to projected depth
            conv2 = tf.layers.conv2d(concat, filters=filters, kernel_size=(1,1), strides=(1,1), padding='SAME')
            conv2 = batch_norm(conv2, filters)
            conv2 = tf.nn.relu(conv2)
        if strides != (1,1):
            input = tf.layers.conv2d(input, filters=filters, kernel_size=(1,1), strides=strides, padding='SAME')
            input = batch_norm(input, filters)

        output = tf.add(input, conv2)
        return tf.nn.relu(output)


train_x, train_y, eval_x, eval_y = get_data()

train_y = np.reshape(train_y, (-1, 1))
eval_y = np.reshape(eval_y, (-1, 1))

training_dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y))
test_dataset = tf.data.Dataset.from_tensor_slices((eval_x, eval_y))

training_dataset = training_dataset.batch(batch_size)
training_dataset = training_dataset.repeat()
test_dataset = test_dataset.batch(10000)

handle = tf.placeholder(tf.string, shape=[])
iterator = tf.data.Iterator.from_string_handle(handle, training_dataset.output_types, training_dataset.output_shapes)
input, label = iterator.get_next()

x = tf.placeholder_with_default(input, shape=[None, 784], name="x")
y = tf.placeholder_with_default(label, shape=[None, 1], name="y")

y_onehot = tf.one_hot(label, depth=10)
y_onehot = tf.reshape(y_onehot, shape=[-1, 10])

input_layer = tf.reshape(x, shape=[-1, 28, 28, 1])
#input_layer output = {BATCH_SIZE, 28, 28, 1]

conv1 = tf.layers.conv2d(input_layer, filters=32, kernel_size=(5, 5), strides=(1, 1), padding="SAME", activation=tf.nn.relu, trainable=True)
#conv1 output = [BATCH_SIZE, 28, 28, 32]
pool1 = tf.layers.max_pooling2d(conv1, pool_size=[2, 2], strides=[2, 2])
#pool1 output = [BATCH_SIZE, 14, 14, 32]

#ResNext Block #1
block1 = res_block("resBlock1", pool1, strides=(1,1), filters=32, cardinality=4, pooling1_1=4)
#[BATCH_SIZE, 14, 14, 32]

#ResNext Block #2
block2 = res_block("resBlock2", block1, strides=(1,1), filters=32, cardinality=4, pooling1_1=4)
#[BATCH_SIZE, 14, 14, 32]

#ResNext Block #3
block3 = res_block("resBlock3", block2, strides=(1,1), filters=32, cardinality=4, pooling1_1=4)
#[BATCH_SIZE, 14, 14, 32]

#ResNext Block #4
block4 = res_block("resBlock4", block3, strides=(2,2), filters=64, cardinality=8, pooling1_1=4)
#[BATCH_SIZE, 7, 7, 64]

#ResNext Block #5
block5 = res_block("resBlock5", block4, strides=(1,1), filters=64, cardinality=8, pooling1_1=4)
#[BATCH_SIZE, 7, 7, 64]

with tf.name_scope("global_average_pooling"):
    avg_pool = tf.reduce_mean(block5, axis=[1, 2])
    #[BATCH_SIZE, 64]

with tf.name_scope("output_layer"):
    out_weights = tf.Variable(tf.truncated_normal(shape=[64, 10], stddev=0.1), tf.float32)
    out_biases = tf.Variable(tf.zeros(shape=[10]), tf.float32)
    output = tf.matmul(avg_pool, out_weights) + out_biases

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=y_onehot))

global_step = tf.Variable(0, trainable=False)
learning_rate = tf.train.exponential_decay(training_rate, global_step, 1000, 0.9, staircase=True)
train = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

correct_prediction = tf.equal(tf.argmax(y_onehot, axis=1), tf.argmax(output, axis=1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

tf.summary.scalar("loss", loss)
tf.summary.scalar("accuracy", accuracy)

merged = tf.summary.merge_all()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    training_handle = sess.run(training_dataset.make_one_shot_iterator().string_handle())
    test_handle = sess.run(test_dataset.make_one_shot_iterator().string_handle())

    train_writer = tf.summary.FileWriter(path, sess.graph)

    for i in range(batches):
        try:
            acc, cost, _, summary = sess.run([accuracy, loss, train, merged], feed_dict={handle: training_handle})
            print("Mini-batch:", i, ", Accuracy:", acc, " Loss:", cost)
            train_writer.add_summary(summary, i)
        except tf.errors.OutOfRangeError:
            print("end of training data")
            break
    final_acc = sess.run([accuracy], feed_dict={handle: test_handle})
    print("Test Accuracy:", final_acc)
