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
import svmtree
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
n          = 5              # Set the depth of the architecture: n = 5 -> 32 layers (See He et al. paper)
nb_val     = 0              # Validation samples per class
nb_cl      = 10             # Classes per group
nb_groups  = int(100/nb_cl)
Num_all_protos =1000        # 保留集所有样本的数量
nb_protos  = 20             # Number of prototypes per class at the end: total protoset memory/ total number of classes
alph       = 5              # 在每个支持向量样本 周边抽取的样本数
epochs     = 70             # Total number of epochs
lr_old     = 0.05             # Initial learning rate
lr_strat   = [1,2]       # Epochs where learning rate gets decreased
lr_factor  = 5.             # Learning rate decrease factor
wght_decay = 0.00001        # Weight Decay
nb_runs    = 1              # 总的执行次数 Number of runs (random ordering of classes at each run)10*10=100类
np.random.seed(1993)        # Fix the random seed
Cifar_train_file, Cifar_test_file, save_path = set_data_path.get_data_path()
################################################################

#loading dataset
print("\n")
# Initialization
dictionary_size   = 500-nb_val
loss_batch        = []
class_means       = np.zeros((128,100,2,nb_groups))
files_protoset     = []
# Select the order for the class learning
order = np.load('./order.npy', encoding='latin1')
for i in range(100):
    files_protoset.append([])
#top1_acc_list_cumul = np.zeros((100/nb_cl,3,nb_runs))
#top1_acc_list_ori   = np.zeros((100/nb_cl,3,nb_runs))

#执行多次.................................
for step_classes in [10]:#,5,10,20,50]:
    nb_cl = step_classes  # Classes per group
    nb_groups = 2#int(100 / nb_cl)
    for itera in range(nb_groups):#100/nb_cl
        if itera == 0:#第一次迭代增加批次 后面网络被初始化 效率提高
            epochs = 3
        else:
            epochs = 3
        """
        1、先构建网络，定义一些变量
        2、构建损失函数
        3、构建循环网络
        4、筛选保留集样本
        5、先实现残差网络　再实现增量学习
        6、实现简单的残差网络
        """
        # Create neural network model
        print('Run {0} starting ...'.format(itera))
        print("Building model and compiling functions...")

        image_train, label_train,image_test, label_test = utils_data.load_data(Cifar_train_file, Cifar_test_file)
        #next batch
        image_batch, label_batch_0, file_xu_batch = utils_data.Prepare_train_data_batch(image_train,label_train,files_protoset,itera,order,nb_cl,batch_size)
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
            images_, label_, file_xu_ = utils_data.GetData(image_train, label_train, files_protoset, itera, order, nb_cl)
            print('training*****************************************************')
            print("Batch of classes {} out of {} batches".format(itera, 100 / nb_cl))
            for epoch in range(epochs):  # 训练模型
                print('Epoch %i' % epoch)
                # print(len(files_from_cl))
                for i in range(int(np.ceil(len(label_)/ batch_size))):  # 5000/128
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
            save_model_path = save_path + 'step_'+str(step_classes)+'_classes'+'/SVM/'
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
            '''
        model_path = save_path + 'step_'+str(nb_cl)+'_classes/SVM/'
        inits, scores, label_batch, loss_class, file_xu_batch, op_feature_map = utils_data.reading_data_and_preparing_network('train',image_train,label_train, files_protoset,itera, batch_size, order,nb_cl, model_path)
        with tf.Session(config=config) as sess:
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            void3 = sess.run(inits)

            # Load the training samples of the current batch of classes in the feature space to apply the herding algorithm
            #完成全部迭代
            Dtot, label_dico,file_process = utils_data.load_class_in_feature_space(nb_cl, batch_size,scores, label_batch, loss_class,
                                                                      file_xu_batch,op_feature_map, sess, len(label_))
            #file_process = np.array([x.decode() for x in file_process])
            # Herding procedure : ranking of the potential exemplars
            print('Exemplars selection starting ...')

            # Herding procedure : ranking of the potential exemplars
            # 参考文献中预测是用 KNN 预测的
            ###########################################
            cl_feature_svm = []  # for SVM trees
            label_for_svm = []
            nu_cl_for_svm = []
            cl_list = []
            num_cl = nb_cl
            # proto_file=files_protoset
            # nb_protos_cl=nb_proto
            nu_cl_itera = nb_cl
            pic_name = []
            for iter_dico in range(nb_cl):
                ind_cl = np.where(label_dico == order[iter_dico + itera * nb_cl])[0]  # ind_cl：序列
                D = Dtot[:, ind_cl]  # 特征//特征是列保存的列号是类别 需要不同类别的特征
                # nothing nothing;
                cl_feature_svm.extend(D.T)  # 转秩为行向量
                label_for_svm.extend([iter_dico + itera * nb_cl] * (len(ind_cl)))#标签 从0开始到100 可以通过order 确定其实际类别
                nu_cl_for_svm.append(len(ind_cl))# 类别数量
                cl_list.append((iter_dico + itera * nb_cl))
                pic_name.extend(file_process[ind_cl])
            coord.request_stop()
            coord.join(threads)
            cl_feature_svm = np.float32(cl_feature_svm)
            label_for_svm = np.int32(label_for_svm)
            # cl_list = np.unique(cl_list)
            cl_list = [int(cl - itera * nb_cl) for cl in cl_list]#又从零开始编号
            print('Exemplars selection starting by SVMT ...')

            with open('cl_feature_svm.pickle', 'wb') as fp:
                cPickle.dump(cl_feature_svm, fp)
            with open('label_for_svm.pickle', 'wb') as fp:
                cPickle.dump(label_for_svm, fp)
            with open('nu_cl_for_svm.pickle', 'wb') as fp:
                cPickle.dump(nu_cl_for_svm, fp)
            with open('cl_list.pickle', 'wb') as fp:
                cPickle.dump(cl_list, fp)
            with open('pic_name.pickle', 'wb') as fp:
                cPickle.dump(pic_name, fp)
            nb_protos_ = np.ceil(Num_all_protos /((itera+1)*num_cl))

            # 更新样本 也许有更好的策略
            if itera>0:
                for i in range(num_cl*itera):# 旧样本
                    class_index = order[itera * num_cl + i]
                    files_protoset[class_index] = files_protoset[class_index][0:nb_protos_]
            protoset_tmp = []
            for i in range(10):
                protoset_tmp.append([])
            svmtree.svm_recursion_fixed_nu_proto(cl_feature_svm, label_for_svm, nu_cl_for_svm, cl_list, num_cl,
                                         files_protoset, itera, nb_protos_, alph, nu_cl_itera, pic_name)
            for i in range(num_cl):
                class_index = order[itera * num_cl + i]
                files_protoset[class_index] = protoset_tmp[i]
            print(files_protoset)
        # Reset the graph
        tf.reset_default_graph()

    with open(str(nb_cl) + 'files_protoset_by'+str(step_classes)+'classes.pickle', 'wb') as fp:
        cPickle.dump(files_protoset, fp)