import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt
# 读取数据
os.environ["CUDA_VISIBLE_DEVICES"] =  "2"
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

def get_files(filename):
    class_train = []
    label_train = []
    for train_class in os.listdir(filename):
        for pic in os.listdir(filename+train_class):
            name = train_class.split(sep='.')
            class_train.append(filename+train_class+'/'+pic)
            label_train.append(name[0])
    temp = np.array([class_train, label_train])
    temp = temp.transpose()
    # shuffle the samples
    np.random.shuffle(temp)
    # after transpose, images is in dimension 0 and label in dimension 1
    # 转换成list因为后面一些tensorflow函数（get_batch）接收的是list格式数据
    image_list = list(temp[:, 0])
    label_list = list(temp[:, 1])
    label_list = [int(i) for i in label_list]
    # print(label_list)
    return image_list, label_list


# 产生用于训练的批次
# 生成相同大小的批次
def get_batch(image, label, image_W, image_H, batch_size, capacity):
    # image, label: 要生成batch的图像和标签list
    # image_W, image_H: 图片的宽高
    # batch_size: 每个batch有多少张图片
    # capacity: 队列容量
    # return: 图像和标签的batch

    # 将python.list类型转换成tf能够识别的格式
    image = tf.cast(image, tf.string)
    label = tf.cast(label, tf.int32)

    # 生成队列
    input_queue = tf.train.slice_input_producer([image, label])

    image_contents = tf.read_file(input_queue[0])
    label = input_queue[1]
    image = tf.image.decode_jpeg(image_contents, channels=3)

    # 统一图片大小
    image = tf.image.resize_images(image, [image_H, image_W], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    image = tf.cast(image, tf.float32)
    # image = tf.image.per_image_standardization(image)   # 标准化数据
    image_batch, label_batch = tf.train.batch([image, label],
                                              batch_size=batch_size,
                                              num_threads=64,  # 线程
                                              capacity=capacity)

    return image_batch, label_batch


TRAINING = tf.Variable(initial_value=True, dtype=tf.bool, trainable=False)


def identity_block(X_input, kernel_size, filters, stage, block):
    """
    Implementation of the identity block as defined in Figure 3
    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    kernel_size -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network
    Returns:
    X -- output of the identity block, tensor of shape (n_H, n_W, n_C)
    """

    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    with tf.name_scope("id_block_stage"+str(stage)):
        filter1, filter2, filter3 = filters
        X_shortcut = X_input

        # First component of main path
        x = tf.layers.conv2d(X_input, filter1,
                 kernel_size=(1, 1), strides=(1, 1),name=conv_name_base+'2a')
        x = tf.layers.batch_normalization(x, axis=3, name=bn_name_base+'2a', training=TRAINING)
        x = tf.nn.relu(x)

        # Second component of main path
        x = tf.layers.conv2d(x, filter2, (kernel_size, kernel_size),
                                 padding='same', name=conv_name_base+'2b')
        # batch_norm2 = tf.layers.batch_normalization(conv2, axis=3, name=bn_name_base+'2b', training=TRAINING)
        x = tf.nn.relu(x)

        # Third component of main path
        x = tf.layers.conv2d(x, filter3, kernel_size=(1, 1),name=conv_name_base+'2c')
        x = tf.layers.batch_normalization(x, axis=3, name=bn_name_base + '2c', training=TRAINING)

        # Final step: Add shortcut value to main path, and pass it through a RELU activation
        X_add_shortcut = tf.add(x, X_shortcut)
        add_result = tf.nn.relu(X_add_shortcut)

    return add_result


def convolutional_block(X_input, kernel_size, filters, stage, block, stride = 2):
    """
    Implementation of the convolutional block as defined in Figure 4
    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    kernel_size -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network
    stride -- Integer, specifying the stride to be used
    Returns:
    X -- output of the convolutional block, tensor of shape (n_H, n_W, n_C)
    """

    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    with tf.name_scope("conv_block_stage" + str(stage)):

        # Retrieve Filters
        filter1, filter2, filter3 = filters

        # Save the input value
        X_shortcut = X_input

        # First component of main path
        x = tf.layers.conv2d(X_input, filter1,
                                 kernel_size=(1, 1),
                                 strides=(stride, stride),
                                 name=conv_name_base+'2a')
        x = tf.layers.batch_normalization(x, axis=3, name=bn_name_base+'2a', training=TRAINING)
        x = tf.nn.relu(x)

        # Second component of main path
        x = tf.layers.conv2d(x, filter2, (kernel_size, kernel_size), name=conv_name_base + '2b',padding='same')
        x = tf.layers.batch_normalization(x, axis=3, name=bn_name_base + '2b', training=TRAINING)
        x = tf.nn.relu(x)

        # Third component of main path
        x = tf.layers.conv2d(x, filter3, (1, 1), name=conv_name_base + '2c')
        x = tf.layers.batch_normalization(x, axis=3, name=bn_name_base + '2c', training=TRAINING)

        # SHORTCUT PATH
        X_shortcut = tf.layers.conv2d(X_shortcut, filter3, (1,1),
                                      strides=(stride, stride), name=conv_name_base + '1')
        X_shortcut = tf.layers.batch_normalization(X_shortcut, axis=3, name=bn_name_base + '1', training=TRAINING)

        # Final step: Add shortcut value to main path, and pass it through a RELU activation
        X_add_shortcut = tf.add(X_shortcut, x)
        add_result = tf.nn.relu(X_add_shortcut)

    return add_result


def ResNet50_reference(x):
    """
    Implementation of the popular ResNet50 the following architecture:
    CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> CONVBLOCK -> IDBLOCK*2 -> CONVBLOCK -> IDBLOCK*3
    -> CONVBLOCK -> IDBLOCK*5 -> CONVBLOCK -> IDBLOCK*2 -> AVGPOOL -> TOPLAYER
    Arguments:
    Returns:
    """


    # stage 1
    x = tf.layers.conv2d(x, filters=64, kernel_size=(7, 7), strides=(2, 2), name='conv1')
    x = tf.layers.batch_normalization(x, axis=3, name='bn_conv1')
    x = tf.nn.relu(x)
    x = tf.layers.max_pooling2d(x, pool_size=(3, 3),strides=(2, 2))


    # stage 2
    x = convolutional_block(x, kernel_size=3, filters=[64, 64, 256], stage=2, block='a', stride=1)
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

    # stage 3
    x = convolutional_block(x, kernel_size=3, filters=[128,128,512],
                                            stage=3, block='a', stride=2)
    x = identity_block(x, 3, [128,128,512], stage=3, block='b')
    x = identity_block(x, 3, [128,128,512], stage=3, block='c')
    x = identity_block(x, 3, [128,128,512], stage=3, block='d')

    # stage 4
    x = convolutional_block(x, kernel_size=3, filters=[256, 256, 1024], stage=4, block='a', stride=2)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

    # stage 5
    x = convolutional_block(x,kernel_size=3,filters=[512, 512, 2048], stage=5, block='a', stride=2)
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

    x = tf.layers.average_pooling2d(x, pool_size=(7, 7), strides=(1,1))

    flatten = tf.layers.flatten(x, name='flatten')
    logits = tf.layers.dense(flatten, units=200, activation=tf.nn.relu)

    return logits


def loss(logits, labels):
    with tf.variable_scope("loss") as scope:
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                       labels=labels, name="xentropy_per_example")
        loss = tf.reduce_mean(cross_entropy, name="loss")
        tf.summary.scalar(scope.name + "loss", loss)
    return loss


def get_accuracy(logits, labels):
    with tf.variable_scope("accuracy") as scope:
        correct = tf.nn.in_top_k(logits, labels, 1)
        correct = tf.cast(correct, tf.float16)
        accuracy = tf.reduce_mean(correct)
        tf.summary.scalar(scope.name + "accuracy", accuracy)
    return accuracy


def trainning(loss, lr):
    train_op = tf.train.GradientDescentOptimizer (lr).minimize(loss)  # 当数据量小时，加入var_list更新指定参数
    return train_op

data_dir = 'D:/v-zzx/train/'
test_dir = 'D:/v-zzx/test/'
log_dir = 'D:/v-zzx/log/'
# data_dir = 'D:/data/birddata/CUB_200_2011/train/'
# test_dir = 'D:/data/birddata/CUB_200_2011/test/'
# log_dir = 'D:/data/birddata/CUB_200_2011/log/'
size = 224
batch_size = 256
capacity = 256
num_classes = 200
learning_rate = 0.001
epoch = 10000
keep_prob = 0.5  # 随机失活概率

x = tf.placeholder(tf.float32, [batch_size, size, size, 3])
y_ = tf.placeholder(tf.int32, [batch_size])

image, label = get_files(data_dir)
image_batches, label_batches = get_batch(image, label, size, size, batch_size, capacity)
timage, tlabel = get_files(test_dir)
timage_batches, tlabel_batches = get_batch(timage, tlabel, size, size, batch_size, capacity)

q = ResNet50_reference(x)
cost = loss(q, y_)
# 衰减的学习率
# global_step = tf.Variable(0, trainable=False)
# starter_learning_rate = 0.001
# learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
#                                            200, 0.96, staircase=True)
train_op = trainning(cost, learning_rate)
acc = get_accuracy(q, y_)
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
saver = tf.train.Saver()
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)
Cross_loss = []
Test_acc = []
Train_acc = []


try:
    for step in np.arange(epoch):
        print(step)
        if coord.should_stop():
            break
        x_1, y_1 = sess.run([image_batches, label_batches])
        _, train_acc, train_loss = sess.run([train_op, acc, cost], feed_dict={x:x_1,y_:y_1})
        Cross_loss.append(train_loss)
        Train_acc.append(train_acc)
        print("loss:{} \ntrain_accuracy:{}".format(train_loss, train_acc))
        x_2, y_2 = sess.run([timage_batches, tlabel_batches])
        # print("test_accuracy:", sess.run(acc, feed_dict={x: x_2, y_:y_2}))
        test_acc = sess.run(acc, feed_dict={x: x_2, y_:y_2})
        Test_acc.append(test_acc)
        print("test_accuracy:{}".format(test_acc))
        if step % 1000 == 0:
            check = os.path.join(log_dir, "model.ckpt")
            saver.save(sess, check, global_step=step)
except tf.errors.OutOfRangeError:
    print("Done!!!")
finally:
    coord.request_stop()
coord.join(threads)
# 画出loss图像

fig, ax = plt.subplots(figsize=(13, 6))
ax.plot(Cross_loss)
plt.grid()
plt.title('Train loss')
plt.show()
fig, ax = plt.subplots(figsize=(13, 6))
ax.plot(Train_acc)
plt.grid()
plt.title('Train acc')
plt.show()
fig, ax = plt.subplots(figsize=(13, 6))
ax.plot(Test_acc)
plt.grid()
plt.title('Test acc')
plt.show()

sess.close()
