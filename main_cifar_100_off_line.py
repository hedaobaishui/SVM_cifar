#硬训练
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

try:
	import cPickle
except:
	import pickle as cPickle

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
##############Incremental Learning Setting######################
gpu        = '0'
batch_size = 100            # Batch size
n          = 5              # Set the depth of the architecture: n = 5 -> 32 layers (See He et al. paper)
nb_val     = 0              # Validation samples per class
nb_cl      = 10             # Classes per group
nb_groups  = int(100/nb_cl)
nb_protos  = 20             # Number of prototypes per class at the end: total protoset memory/ total number of classes
epochs     = 50             # Total number of epochs
lr_old     = 0.05             # Initial learning rate
lr_strat   = [30,40]       # Epochs where learning rate gets decreased
lr_factor  = 5.             # Learning rate decrease factor
wght_decay = 0.00001        # Weight Decay
nb_runs    = 1              # 总的执行次数 Number of runs (random ordering of classes at each run)10*10=100类
np.random.seed(1993)        # Fix the random seed
Cifar_train_file  = 'F:/Dataset/ILSVRC2012/cifar-100-python/train'
#需要修改
Cifar_test_file   = 'F:/Dataset/ILSVRC2012/cifar-100-python/test'#需要修改
save_path         = './model/'
################################################################

#loading dataset
print("\n")
# Initialization
dictionary_size   = 500-nb_val
loss_batch        = []
class_means       = np.zeros((128,100,2,nb_groups))
files_protoset     = []
save_weights      =  None
# Select the order for the class learning
# 使用固定的order
order = np.load('./order.npy', encoding='latin1')

for i in range(100):
    files_protoset.append([])
#top1_acc_list_cumul = np.zeros((100/nb_cl,3,nb_runs))
#top1_acc_list_ori   = np.zeros((100/nb_cl,3,nb_runs))

#执行多次.................................
for step_classes in [10,20,50]:#2,5
    nb_cl = step_classes
    nb_groups = int(100/nb_cl)
    for itera in range(nb_groups):#100/nb_cl
        """
        1、先构建网络，定义一些变量
        2、构建损失函数
        3、构建循环网络
        """
        # Create neural network model
        print('Run {0} starting ...'.format(itera))
        print("Building model and compiling functions...")

        image_train, label_train,image_test, label_test = utils_data.load_data(Cifar_train_file, Cifar_test_file)
        #next batch
        #获得当前为止所有类别的数据
        image_batch, label_batch_0, file_protoset_batch = utils_data.Prepare_train_data_batch_all(image_train,label_train,files_protoset,itera,order,nb_cl,batch_size)
        label_batch = tf.one_hot(label_batch_0, 100)
        variables_graph, variables_graph2, scores, scores_stored = utils_cifar.prepareNetwork(gpu,image_batch)
        with tf.device('/gpu:0'):
            scores        = tf.concat(scores,0)
            l2_reg        = wght_decay * tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, scope='ResNet34'))
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

            if itera > 0:
                void0 = sess.run([(variables_graph[i]).assign(save_weights[i]) for i in range(len(variables_graph))])
                #void1 = sess.run(op_assign)

            print('training*****************************************************')
            print("Batch of classes {} out of {} batches".format(itera, 100 / nb_cl))
            for epoch in range(epochs):  # 训练模型
                print('Epoch %i' % epoch)
                # print(len(files_from_cl))
                for i in range(int(np.ceil(500*nb_cl/ batch_size))):  # 5000/128
                    loss_class_val, _, sc, lab = sess.run([loss_class, train_step, scores, label_batch_0],
                                                          feed_dict={learning_rate: lr})
                    loss_batch.append(loss_class_val)

                    # Plot the training error every 10 batches
                    if len(loss_batch) == 10:
                        print("Training error:")
                        print(np.mean(loss_batch))
                        loss_batch = []

                    # Plot the training top 1 accuracy every 80 batches
                    # print('i=', i)
                    if (i + 1) % 20 == 0:
                        stat = []
                        stat += ([ll in best for ll, best in zip(lab, np.argsort(sc, axis=1)[:, -1:])])# Top 1 正确率
                        stat = np.average(stat)
                        print('Training accuracy %f' % stat)

                # Decrease the learning by 5 every 10 epoch after 20 epochs at the first learning rate
                if epoch in lr_strat:
                    lr /= lr_factor

            coord.request_stop()
            coord.join(threads)

            # copy weights to store network
            print('saving model')
            save_weights = sess.run([variables_graph[i] for i in range(len(variables_graph))])
            save_model_path = save_path + 'step_'+str(step_classes)+'_classes'+'/off_line/'
            utils_cifar.save_model(save_model_path + 'model-iteration' + str(nb_cl) + '-%i.pickle' % itera, scope='ResNet34',
                                    sess=sess)
        tf.reset_default_graph()