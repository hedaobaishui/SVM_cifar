import tensorflow as tf
import pickle
import utils_cifar
import sys
import os
import time
import string
import random
import numpy as np
import utils_data

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
##############Incremental Learning Setting######################
gpu        = '0'
batch_size = 128            # Batch size
n          = 5              # Set the depth of the architecture: n = 5 -> 32 layers (See He et al. paper)
nb_val     = 0              # Validation samples per class
nb_cl      = 10             # Classes per group 
nb_protos  = 20             # Number of prototypes per class at the end: total protoset memory/ total number of classes
epochs     = 70             # Total number of epochs 
lr_old     = 2.             # Initial learning rate
lr_strat   = [49, 63]       # Epochs where learning rate gets decreased
lr_factor  = 5.             # Learning rate decrease factor
wght_decay = 0.00001        # Weight Decay
nb_runs    = 1              # 总的执行次数 Number of runs (random ordering of classes at each run)10*10=100类
np.random.seed(1993)        # Fix the random seed
Cifar_train_file   = 'F:\Dataset\ILSVRC2012\cifar-100-python/train'
#需要修改
Cifar_test_file   = 'F:\Dataset\ILSVRC2012\cifar-100-python/test'#需要修改
################################################################

#loading dataset
print("\n")
# Initialization
dictionary_size     = 500-nb_val
#top1_acc_list_cumul = np.zeros((100/nb_cl,3,nb_runs))
#top1_acc_list_ori   = np.zeros((100/nb_cl,3,nb_runs))

#执行多次.................................
for iteration_run in range(nb_runs):
    """
    1、先构建网络，定义一些变量
    2、构建损失函数
    3、构建循环网络
    4、筛选保留集样本
    5、先实现残差网络　再实现增量学习
    6、实现简单的残差网络
    """
    # Select the order for the class learning 
    order = np.arange(100)
    np.random.shuffle(order)
    np.save('order', order)#### 存储样本顺序的序列

    # Create neural network model
    print('Run {0} starting ...'.format(iteration_run))
    print("Building model and compiling functions...")
    image_train, label_train,image_test, label_test = utils_data.load_data(Cifar_train_file, Cifar_test_file)
    #next batch
    image_batch, label_batch_0 = utils_data.Prepare_train_data_batch(image_train,label_train,nb_runs,order,nb_cl,batch_size)
    label_batch = tf.one_hot(label_batch_0, 100)
    variables_graph, variables_graph2, scores, scores_stored = utils_cifar.prepareNetwork(gpu,image_batch)
    with tf.device('/gpu:0'):
        scores        = tf.concat(scores,0)
        l2_reg        = wght_decay * tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, scope='ResNet18'))
        loss_class    = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=label_batch, logits=scores))
        loss          = loss_class + l2_reg
        learning_rate = tf.placeholder(tf.float32, shape=[])
        opt           = tf.train.MomentumOptimizer(learning_rate, 0.9)#需要修改下下
        train_step    = opt.minimize(loss,var_list=variables_graph)
    with tf.Session(config=config) as sess:
        #Launch the data reader
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        sess.run(tf.global_variables_initializer())
        lr = lr_old

        #training*****************************************************
        for epoch in range(epochs):  # 训练模型
            print("Batch of classes {} out of {} batches".format(1, 1))
            print('Epoch %i' % epoch)
            print('file_num_cl=', 100)
            # print(len(files_from_cl))
            for i in range(int(np.ceil(100/ batch_size))):  # 6250/128
                loss_class_val, _, sc, lab = sess.run([loss_class, train_step, scores, label_batch_0],
                                                      feed_dict={learning_rate: lr})
                loss_batch.append(loss_class_val)

                # Plot the training error every 10 batches
                if len(loss_batch) == 10:
                    print("Training error:")
                    print(np.mean(loss_batch))
                    loss_batch = []

                # Plot the training top 1 accuracy every 80 batches
                print('i=', i)
                if (i + 1) % 20 == 0:
                    stat = []
                    stat += ([ll in best for ll, best in zip(lab, np.argsort(sc, axis=1)[:, -1:])])
                    stat = np.average(stat)
                    print('Training accuracy %f' % stat)

            # Decrease the learning by 5 every 10 epoch after 20 epochs at the first learning rate
            if epoch in lr_strat:
                lr /= lr_factor

        coord.request_stop()
        coord.join(threads)

        # copy weights to store network
        save_weights = sess.run([variables_graph[i] for i in range(len(variables_graph))])
        utils_cifar.save_model('./' + 'model-iteration' + str(nb_cl) + '-%i.pickle' % itera, scope='ResNet18',
                                sess=sess)

