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
batch_size = 128            # Batch size
n          = 5              # Set the depth of the architecture: n = 5 -> 32 layers (See He et al. paper)
nb_val     = 0              # Validation samples per class
nb_cl      = 10             # Classes per group
nb_groups  = int(100/nb_cl)
nb_protos  = 20             # Number of prototypes per class at the end: total protoset memory/ total number of classes
epochs     = 70             # Total number of epochs
lr_old     = 2.             # Initial learning rate
lr_strat   = [49, 63]       # Epochs where learning rate gets decreased
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

for i in range(100):
    files_protoset.append([])
#top1_acc_list_cumul = np.zeros((100/nb_cl,3,nb_runs))
#top1_acc_list_ori   = np.zeros((100/nb_cl,3,nb_runs))

#执行多次.................................
for step_classes in [2,5,10,20,50]:
    nb_cl = step_classes  # Classes per group
    nb_groups = int(100 / nb_cl)
    for itera in range(nb_groups):#100/nb_cl
        if itera == 0:#第一次迭代增加批次 后面网络被初始化 效率提高
            epochs = 80
        else:
            epochs = 50
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
            save_model_path = save_path + 'step_'+str(step_classes)+'_classes'+'/NCM/'
            utils_cifar.save_model('' + 'model-iteration' + str(nb_cl) + '-%i.pickle' % itera, scope='ResNet34',
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
        inits, scores, label_batch, loss_class, file_string_batch, op_feature_map = utils_data.reading_data_and_preparing_network('train',image_train,label_train, files_protoset,itera, batch_size, order,nb_cl, save_path)
        with tf.Session(config=config) as sess:
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            void3 = sess.run(inits)

            # Load the training samples of the current batch of classes in the feature space to apply the herding algorithm
            Dtot, label_dico,file_process = utils_data.load_class_in_feature_space(nb_cl, batch_size,scores, label_batch, loss_class,
                                                                      file_string_batch,op_feature_map, sess)
            file_process = np.array([x.decode() for x in file_process])
            # Herding procedure : ranking of the potential exemplars
            print('Exemplars selection starting ...')
            for iter_dico in range(nb_cl):
                ind_cl = np.where(label_dico == order[iter_dico + itera * nb_cl])[0]
                D = Dtot[:, ind_cl]
                files_iter = file_process[ind_cl]
                mu = np.mean(D, axis=1)
                w_t = mu
                step_t = 0
                while not(len(files_protoset[itera*nb_cl+iter_dico]) == nb_protos_cl) and step_t<1.1*nb_protos_cl:
                    tmp_t = np.dot(w_t, D) #一维数组 内积  二维数组：矩阵积
                    ind_max = np.argmax(tmp_t) #取出数组最大值的索引
                    w_t = w_t + mu - D[:, ind_max] #
                    step_t += 1
                    if files_iter[ind_max] not in files_protoset[itera * nb_cl + iter_dico]:#这里要添加的是样本的序号
                        files_protoset[itera * nb_cl + iter_dico].append(files_iter[ind_max])
                        #存储样本名 还是样本数据

            coord.request_stop()
            coord.join(threads)

            # Reset the graph
        tf.reset_default_graph()

        # Class means for iCaRL and NCM
        # class_means 用于分类测试
        print('Computing theoretical class means for NCM and mean-of-exemplars for iCaRL ...')
        for iteration2 in range(itera + 1):
            inits, scores, label_batch, loss_class, file_string_batch, op_feature_map = utils_data.reading_data_and_preparing_network(
                'train',image_train, label_train, files_protoset, itera, batch_size, order, nb_cl, save_path)

            with tf.Session(config=config) as sess:
                coord = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(coord=coord)
                void2 = sess.run(inits)

                Dtot, label_dico, file_process = utils_data.load_class_in_feature_space(nb_cl, batch_size, scores, label_batch,
                                                                          loss_class,
                                                                          file_string_batch,op_feature_map, sess)
                file_process = np.array([x.decode() for x in file_process])
                for iter_dico in range(nb_cl):
                    ind_cl = np.where(label_dico == order[iter_dico + iteration2 * nb_cl])[0]
                    D = Dtot[:, ind_cl]
                    files_iter = file_process[ind_cl]
                    current_cl = order[range(iteration2 * nb_cl, (iteration2 + 1) * nb_cl)]#nb_groups

                    # Normal NCM mean
                    # 各维度信息： 特征：类别：(0,1)：nb_groups
                    class_means[:, order[iteration2 * nb_cl + iter_dico], 1, itera] = np.mean(D, axis=1)
                    # 归一化后的信息
                    class_means[:, order[iteration2 * nb_cl + iter_dico], 1, itera] /= np.linalg.norm(
                        class_means[:, order[iteration2 * nb_cl + iter_dico], 1, itera])

                    # iCaRL approximated mean (mean-of-exemplars)
                    # use only the first exemplars of the old classes:
                    # nb_protos_cl controls the number of exemplars per class
                    ind_herding = np.array(
                        [np.where(files_iter == files_protoset[iteration2 * nb_cl + iter_dico][i])[0][0] for i in
                         range(min(nb_protos_cl, len(files_protoset[iteration2 * nb_cl + iter_dico])))])
                    D_tmp = D[:, ind_herding]
                    class_means[:, order[iteration2 * nb_cl + iter_dico], 0, itera] = np.mean(D_tmp, axis=1)
                    class_means[:, order[iteration2 * nb_cl + iter_dico], 0, itera] /= np.linalg.norm(
                        class_means[:, order[iteration2 * nb_cl + iter_dico], 0, itera])

                coord.request_stop()
                coord.join(threads)

        # Reset the graph
        tf.reset_default_graph()

        # Pickle class means and protoset
        # 每个增量阶段的class_means 不相同
        with open(str(nb_cl) + 'class_means'+str(itera)+'.pickle', 'wb') as fp:
            cPickle.dump(class_means, fp)
        with open(str(nb_cl) + 'files_protoset.pickle', 'wb') as fp:
            cPickle.dump(files_protoset, fp)