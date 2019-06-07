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
train_path = '/home/magic/project/virtualenv/TF/cifar-100-python/train'
test_path = '/home/magic/project/virtualenv/TF/cifar-100-python/test'
save_path = '/home/magic/project/virtualenv/TF/SVM_cifar/model/'#修改为合适的路径

###########################

# Load ResNet settings
# str_mixing = str(nb_cl) + 'mixing.pickle'
# with open(str_mixing, 'rb') as fp:
#     mixing = cPickle.load(fp)

str_settings_resnet = str(nb_cl) + 'settings_resnet.pickle'

# with open(str_settings_resnet, 'rb') as fp:
#     order = cPickle.load(fp)
#     files_valid = cPickle.load(fp)
#     files_train = cPickle.load(fp)

order = np.load('./order.npy', encoding='latin1')

image_train, label_train,image_test, label_test = utils_data.load_data(train_path, test_path)

for nb_cl in [2, 5, 10, 20, 50]:  # 不同类别数量/批次
    nb_groups = int(100 / nb_cl)
    for itera in range(nb_groups):  # 增量学习的次数(迭代产生的模型数量)
        image_batch, label_batch_0 = utils_data.Prepare_test_data_batch(image_test, label_test, itera, order, nb_cl,
                                                                        batch_size)
        label_batch = tf.one_hot(label_batch_0, 100)

        # Load class means
        str_class_means = save_path+str(nb_cl) + 'class_means'+str(itera)+'.pickle'
        with open(str_class_means, 'rb') as fp:
            class_means = cPickle.load(fp)

        # Initialization
        acc_list = np.zeros((nb_groups, 3))
        print("Processing network after {} increments\t".format(itera))
        # Evaluation on cumul of classes or original classes
        if is_cumul == 'cumul':
            eval_groups = np.array(range(itera + 1))
        else:
            eval_groups = [0]

        print("Evaluation on batches {} \t".format(eval_groups))
        # 获得训练好的模型。
        inits, scores, label_batch, loss_class, file_string_batch, op_feature_map = utils_data.reading_data_and_preparing_network(
            'test', image_test, label_test, itera, batch_size, order, nb_cl, save_path)

        with tf.Session(config=config) as sess:
            # Launch the prefetch system
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            sess.run(inits)

            # Evaluation routine
            stat_hb1 = []
            stat_icarl = []
            stat_ncm = []

            for i in range(int(nb_cl*(itera+1)*100 / batch_size)):#迭代次数
                sc, l, loss, files_tmp, feat_map_tmp = sess.run(
                    [scores, label_batch, loss_class, file_string_batch, op_feature_map])
                mapped_prototypes = feat_map_tmp[:, 0, 0, :]
                pred_inter = (mapped_prototypes.T) / np.linalg.norm(mapped_prototypes.T, axis=0)
                sqd_icarl = -cdist(class_means[:, :, 0, itera].T, pred_inter.T, 'sqeuclidean').T
                sqd_ncm = -cdist(class_means[:, :, 1, itera].T, pred_inter.T, 'sqeuclidean').T
                stat_hb1 += ([ll in best for ll, best in zip(l, np.argsort(sc, axis=1)[:, -top:])])
                stat_icarl += ([ll in best for ll, best in zip(l, np.argsort(sqd_icarl, axis=1)[:, -top:])])
                stat_ncm += ([ll in best for ll, best in zip(l, np.argsort(sqd_ncm, axis=1)[:, -top:])])

            coord.request_stop()
            coord.join(threads)

        print('Increment: %i' % itera)
        print('Hybrid 1 top ' + str(top) + ' accuracy: %f' % np.average(stat_hb1))
        print('iCaRL top ' + str(top) + ' accuracy: %f' % np.average(stat_icarl))
        print('NCM top ' + str(top) + ' accuracy: %f' % np.average(stat_ncm))
        acc_list[itera, 0] = np.average(stat_icarl)
        acc_list[itera, 1] = np.average(stat_hb1)
        acc_list[itera, 2] = np.average(stat_ncm)

        # Reset the graph to compute the numbers ater the next increment
        tf.reset_default_graph()

    np.save('results_top' + str(top) + '_acc_' + is_cumul + '_cl' + str(nb_cl), acc_list)#产生5个测试结果。