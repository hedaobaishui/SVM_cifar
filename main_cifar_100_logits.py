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
import set_data_path
try:
	import cPickle
except:
	import pickle as cPickle

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
##############Incremental Learning Setting######################
gpu        = '0'
batch_size = 128            # Batch size
logit_thta = 0.2            # 分类错误的样本占分类正确样本的数量
n          = 5              # Set the depth of the architecture: n = 5 -> 32 layers (See He et al. paper)
nb_val     = 0              # Validation samples per class
nb_cl      = 10             # Classes per group
nb_groups  = int(100/nb_cl)
nb_protos  = 10             # Number of prototypes per class at the end: total protoset memory/ total number of classes
epochs     = 50             # Total number of epochs
lr_old     = 0.05             # Initial learning rate
lr_strat   = [30, 40]       # Epochs where learning rate gets decreased
lr_factor  = 5.             # Learning rate decrease factor
wght_decay = 0.00001        # Weight Decay
nb_runs    = 1              # 总的执行次数 Number of runs (random ordering of classes at each run)10*10=100类
np.random.seed(1993)        # Fix the random seed

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
Cifar_train_file, Cifar_test_file, save_path = set_data_path.get_data_path()

################################################################

#loading dataset
print("\n")
# Initialization
dictionary_size   = 500-nb_val
loss_batch        = []
class_means       = np.zeros((128,100,2,nb_groups))
files_protoset     = []

for i in range(100):
    files_protoset.append([])
#top1_acc_list_cumul = np.zeros((100/nb_cl,3,nb_runs))
#top1_acc_list_ori   = np.zeros((100/nb_cl,3,nb_runs))

#执行多次.................................
for step_classes in [2,10]:#5,20,50]:
    save_model_path = save_path + 'step_' + str(step_classes) + '_classes' + '/logits/'
    nb_cl = step_classes  # Classes per group
    nb_groups = int(100 / nb_cl)
    for itera in range(nb_groups):#100/nb_cl
        if itera == 0:#第一次迭代增加批次 后面网络被初始化 效率提高
            epochs = 1
        else:
            epochs = 1
        """
        1、先构建网络，定义一些变量
        2、构建损失函数
        3、构建循环网络
        4、筛选保留集样本
        5、先实现残差网络　再实现增量学习
        6、实现简单的残差网络
        """
        # Select the order for the class learning
        order = np.load('./order.npy',encoding='latin1')

        # Create neural network model
        print('Run {0} starting ...'.format(itera))
        print("Building model and compiling functions...")

        image_train, label_train,image_test, label_test = utils_data.load_data(Cifar_train_file, Cifar_test_file)
        #next batch
        image_batch, label_batch_0, file_protoset_batch = utils_data.Prepare_train_data_batch(image_train,label_train,files_protoset,itera,order,nb_cl,batch_size)
        label_batch = tf.one_hot(label_batch_0, 100)
        #初次训练
        if itera == 0:
            #不需要蒸馏
            variables_graph, variables_graph2, scores, scores_stored = utils_cifar.prepareNetwork(gpu,image_batch)
            with tf.device('/gpu:0'):
                scores        = tf.concat(scores,0)
                l2_reg        = wght_decay * tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, scope='ResNet34'))
                loss_class    = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=label_batch, logits=scores))
                loss          = loss_class + l2_reg
                learning_rate = tf.placeholder(tf.float32, shape=[])
                opt           = tf.train.MomentumOptimizer(learning_rate, 0.9)#需要修改下下
                train_step    = opt.minimize(loss,var_list=variables_graph)
        elif itera >0:
            #知识蒸馏
            variables_graph, variables_graph2, scores, scores_stored = utils_cifar.prepareNetwork(gpu, image_batch)
            #将上一次网络的输出作为软标签
            op_assign = [(variables_graph2[i]).assign(variables_graph[i]) for i in range(len(variables_graph))]
            with tf.device('/gpu:0'):
                scores = tf.concat(scores, 0)
                scores_stored = tf.concat(scores_stored, 0)
                old_cl = (order[range(itera * nb_cl)]).astype(np.int32)
                new_cl = (order[range(itera * nb_cl, nb_groups * nb_cl)]).astype(np.int32) # ？￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥
                label_old_classes = tf.sigmoid(tf.stack([scores_stored[:, i] for i in old_cl], axis=1))
                label_new_classes = tf.stack([label_batch[:, i] for i in new_cl], axis=1)
                pred_old_classes = tf.stack([scores[:, i] for i in old_cl], axis=1)
                pred_new_classes = tf.stack([scores[:, i] for i in new_cl], axis=1)
                l2_reg = wght_decay * tf.reduce_sum(
                    tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, scope='ResNet34'))
                loss_class = tf.reduce_mean(tf.concat(
                    [tf.nn.sigmoid_cross_entropy_with_logits(labels=label_old_classes, logits=pred_old_classes),
                     tf.nn.sigmoid_cross_entropy_with_logits(labels=label_new_classes, logits=pred_new_classes)], 1))
                loss = loss_class + l2_reg
                learning_rate = tf.placeholder(tf.float32, shape=[])
                opt = tf.train.MomentumOptimizer(learning_rate, 0.9)
                train_step = opt.minimize(loss, var_list=variables_graph)

        with tf.Session(config=config) as sess:
            #Launch the data reader
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            sess.run(tf.global_variables_initializer())
            lr = lr_old

            # Run the loading of the weights for the learning network and the copy network
            if itera > 0:
                void0 = sess.run([(variables_graph[i]).assign(save_weights[i]) for i in range(len(variables_graph))])
                void1 = sess.run(op_assign)

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
                        stat += ([ll in best for ll, best in zip(lab, np.argsort(sc, axis=1)[:, -1:])])
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

            utils_cifar.save_model(save_model_path + 'model-iteration' + str(nb_cl) + '-%i.pickle' % itera, scope='ResNet34',
                                    sess=sess)

        # Reset the graph
        tf.reset_default_graph()
        #筛选保留集
        #总计2000个保留样本 每一类的保留数量（2000/类别数）
        nb_protos_cl = int(np.ceil(nb_protos*100./nb_cl/(itera+1)))
        print('updating reserved file')

        '''
        1.加载训练好的模型参数
        2.用模型对训练数据参数特征
        3.使用数据特征作为依据 进行样本选择
        4.
            '''

        inits, scores, label_batch, loss_class, file_string_batch, op_feature_map = utils_data.reading_data_and_preparing_network('train',image_train,label_train, files_protoset,itera, batch_size, order,nb_cl, save_model_path)
        with tf.Session(config=config) as sess:
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            void3 = sess.run(inits)
            if itera == 0:
                file_num = int(nb_cl * 500)
            else:
                file_num = int(nb_protos * 100 + nb_cl * 500)
            # Load the training samples of the current batch of classes in the feature space to apply the herding algorithm
            Logits, label_dico,file_process = utils_data.load_class_in_feature_space_logits(nb_cl, batch_size,scores, label_batch, loss_class,
                                                                      file_string_batch,op_feature_map, sess,file_num)
            #file_process = np.array([x.decode() for x in file_process])
            # Herding procedure : ranking of the potential exemplars
            print('Exemplars selection starting ...')
            for iter_dico in range(nb_cl):
                ind_cl = np.where(label_dico == order[iter_dico + itera * nb_cl])[0]
                logit = Logits[ind_cl,:]
                label = label_dico[ind_cl]
                # file_now = file_process[ind_cl]
                #Top5 分类正确
                label_zip = zip(label, np.argsort(logit, axis=1)[:, -5:])

                #分类正确的样本的索引
                ind_get = ind_cl[[ll in best for ll,best in label_zip]]
                logit = logit[ind_get,:]#分类正确的样本的 score
                #计算logit 方差 选择方差大的样本添加。
                ind_last = np.argsort([logit_.var() for logit_ in logit])
                #ind_get = ([ll in best for ll, best in zip(label, np.argsort(logit, axis=1)[:, -5:])])
                front_num = int(np.ceil(nb_protos_cl*(1-logit_thta)))
                back_num = int(np.floor(nb_protos_cl*logit_thta))
                files_protoset[order[itera * nb_cl + iter_dico]] = file_process[ind_last[-front_num:]]
                files_protoset[order[itera * nb_cl + iter_dico]] = np.concatenate(files_protoset[order[itera * nb_cl + iter_dico]],[file_process[ind_last[back_num:]]])
            coord.request_stop()
            coord.join(threads)

            # Reset the graph
        tf.reset_default_graph()
        with open(save_model_path+str(nb_cl) + 'files_protoset.pickle', 'wb') as fp:
            cPickle.dump(files_protoset, fp)