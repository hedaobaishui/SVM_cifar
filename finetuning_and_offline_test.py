#模型测试
#实用于微调模型 和离线训练模型
import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
import numpy as np
import scipy
import os
from scipy.spatial.distance import cdist
import scipy.io
import sys

try:
    import cPickle
except:
    import _pickle as cPickle
# Syspath for the folder with the utils files
# sys.path.insert(0, "/data/sylvestre")

import utils_data
import utils_cifar

######### Modifiable Settings ##########
batch_size = 128  # Batch size
nb_cl = 100  # Classes per group
nb_groups = 10  # Number of groups
top = 5  # Choose to evaluate the top X accuracy
is_cumul = 'cumul'  # Evaluate on the cumul of classes if 'cumul', otherwise on the first classes
gpu = '0'  # Used GPU
########################################

######### Paths  ##########
# Working station
train_path = 'F:/Dataset/ILSVRC2012/cifar-100-python/train'
test_path = 'F:/Dataset/ILSVRC2012/cifar-100-python/test'
save_path = './model/'

###########################

str_settings_resnet = str(nb_cl) + 'settings_resnet.pickle'
# with open(str_settings_resnet, 'rb') as fp:
#     order = cPickle.load(fp)
#     files_valid = cPickle.load(fp)
#     files_train = cPickle.load(fp)

order = np.load('./order.npy', encoding='latin1')

image_train, label_train,image_test, label_test = utils_data.load_data(train_path, test_path)

for nb_cl in [2]:#, 5, 10, 20, 50]:  # 不同类别数量/批次
    nb_groups = int(100 / nb_cl)
    acc_list = np.zeros((nb_groups, 1))
    for itera in range(nb_groups):  # 增量学习的次数(迭代产生的模型数量)
        # next batch
        image_batch, label_batch_0, file_protoset_batch= utils_data.Prepare_test_data_batch(image_test, label_test, itera, order, nb_cl,
                                                                        batch_size)
        label_batch = tf.one_hot(label_batch_0, 100)

        # Initialization

        print("Processing network after {} increments\t".format(itera))
        # Evaluation on cumul(累加) of classes or original classes
        if is_cumul == 'cumul':
            eval_groups = np.array(range(itera + 1))
        else:
            eval_groups = [0]

        print("Evaluation on batches {} \t".format(eval_groups))

        #获得训练好的模型。
        model_path = save_path + 'step_'+str(nb_cl)+'_classes/finetuning/'
        inits, scores, label_batch, loss_class, file_string_batch, op_feature_map = utils_data.reading_data_and_preparing_network(
            'test',image_test, label_test,file_protoset_batch, itera, batch_size, order, nb_cl, model_path)

        with tf.Session(config=config) as sess:
            # Launch the prefetch system
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            sess.run(inits)

            # Evaluation routine
            stat_finetuning = []

            for i in range(int(nb_cl*(itera+1)*100 / batch_size)):#迭代次数
                sc, l, loss, files_tmp, feat_map_tmp = sess.run(
                    [scores, label_batch, loss_class, file_string_batch, op_feature_map])
                mapped_prototypes = feat_map_tmp[:, 0, 0, :]

                pred_inter = (mapped_prototypes.T) / np.linalg.norm(mapped_prototypes.T, axis=0)
                stat_finetuning += ([ll in best for ll, best in zip(l, np.argsort(sc, axis=1)[:, -top:])])

            coord.request_stop()
            coord.join(threads)

        print('Increment: %i' % itera)
        print('stat_finetuning  top ' + str(top) + ' accuracy: %f' % np.average(stat_finetuning))
        acc_list[itera, 0] = np.average(stat_finetuning)

        # Reset the graph to compute the numbers ater the next increment
        tf.reset_default_graph()

    np.save('results_top' + str(top) + '_acc_' + is_cumul + '_cl' + str(nb_cl), acc_list)#产生5个测试结果。