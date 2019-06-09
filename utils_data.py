import numpy as np
import os
import scipy.io
import sys
import tensorflow as tf
import utils_cifar
try:
	import cPickle
except:
	import pickle as cPickle
'''
字典形式的数据：
cifar100 data content: 
    { 
    "data" : [(R,G,B, R,G,B ,....),(R,G,B, R,G,B, ...),...]    # 50000张图片，每张： 32 * 32 * 3
    "coarse_labels":[0,...,19],                         # 0~19 super category 
    "filenames":["volcano_s_000012.png",...],   # 文件名
    "batch_label":"", 
    "fine_labels":[0,1...99]          # 0~99 category 
    }
'''
def unpickle(Cifar_file):
	with open(Cifar_file,mode='rb') as file:
		u = cPickle._Unpickler(file)
		u.encoding = 'latin1'
		dict = u.load()
	return dict
def load_data(Cifar_train_file,Cifar_test_file,nb_per_cl_val=0):
	xs=[]
	ys=[]
	for j in range(1):
		d = unpickle(Cifar_train_file)
		x = d['data']#训练数据。
		y = d['fine_labels']#标签
		xs.append(x)
		ys.append(y)
	
	d = unpickle(Cifar_test_file)
	xs.append(d['data'])
	ys.append(d['fine_labels'])
	
	#求解图像均值
	x = np.concatenate(xs)/np.float32(255)#将xs连接起来 #归一化 2个元素
	y = np.concatenate(ys)
	
	x = np.dstack((x[:,:1024],x[:,1024:2048],x[:,2048:]))#原数据1*1024*3（RGB）
	x = x.reshape((x.shape[0],32,32,3)).transpose(0,1,2,3)#后三个位置的坐标转换 变成 5000*3*32*32
	#60000* 3072# # # #  #60000* 1024*3# 	#减去每个像素的均值
	pixel_mean = np.mean(x[0:5000],axis=0)
	x -=pixel_mean

	#创建训练集
	train_sample_cl=500-nb_per_cl_val
	X_train = np.zeros((train_sample_cl*100,32,32,3))
	Y_train = np.zeros(train_sample_cl*100)
	X_vaild = np.zeros((nb_per_cl_val*100,32,32,3))
	Y_vaild = np.zeros(nb_per_cl_val*100)
	for i in range(100):#一次性将所有样本都分配好了
		index_y=np.where(y[0:50000]==i)
		np.random.shuffle(index_y)
		X_train[i*train_sample_cl:(i+1)*train_sample_cl]=x[index_y[0:train_sample_cl],:,:,:]
		Y_train[i*train_sample_cl:(i+1)*train_sample_cl]=y[index_y[0:train_sample_cl]]
		# X_vaild[i*nb_per_cl_val:(i+1)*nb_per_cl_val]=x[index_y[train_sample_cl:500],:,:,:]
		# Y_vaild[i*nb_per_cl_val:(i+1)*nb_per_cl_val]=y[index_y[train_sample_cl:500]]
		
	X_test = x[50000:,:,:,:]
	Y_test = y[50000:].astype(int)
	Y_train = Y_train.astype(int)
	#这里数据都有了将数据转化为tensor
	#需要将数据转化为tensorflow的数据 还要考虑batch_size。
	return X_train,Y_train,X_test,Y_test
	# return dict(
	# X_train = X_train,
	# Y_train = Y_train,
	# X_test = X_test,
	# Y_test = Y_test,
	# )

'''
#函数名 : GetData prepare_train_data_batch
#作者:magic
#日期:2019.4.19
#作用:准备可以用来训练模型的数据(包括保留集的数据)
#参数:数据集,对应样本的标签,batch_size
#返回:返回准备好的数据集和对应的标签
'''
#nu_runs:第几次增量学习#nb_cl每次迭代的类别数
def GetData(train_data,train_data_label,xu_protoset,itera,order,nb_cl):
	traindata_index = order[itera*nb_cl:(itera+1)*nb_cl]
	images =[]
	label =[]
	file_xu=[]
	for i in traindata_index:
		index = np.where(train_data_label[0:50000] == i)
		images.append(train_data[index])
		label.append(train_data_label[index])
		file_xu.extend(index)
	#添加保留集的数据信息
	# xu_protoset[1]=[1,2,3,4,5]
	# xu_protoset[2] = [2021, 2022,2023, 2024, 2025]
	for i in range(100):
		if len(xu_protoset[i]) is not 0:
			# 添加图像信息
			images.append(train_data[xu_protoset[i]])
			# 添加图像标签
			label.append(train_data_label[xu_protoset[i]])
			file_xu.append(xu_protoset[i])
		else:
			continue
	file_xu = np.concatenate(file_xu)
	images  = np.concatenate(images)
	label   = np.concatenate(label)
	return images,label,file_xu
def Prepare_train_data_batch(train_data,train_data_label,xu_protoset,itera,order,nb_cl,batch_size=128):
	images, label, file_xu = GetData(train_data,train_data_label,xu_protoset,itera,order,nb_cl)
	images = tf.cast(images, tf.float32)
	label = tf.cast(label, tf.int32)
	file_xu = tf.cast(file_xu, tf.int32)
	# 从tensor列表中按顺序或随机抽取一个tensor
	input_queue = tf.train.slice_input_producer([images, label, file_xu], shuffle=True)
	image_batch, label_batch,file_xu_batch = tf.train.batch(input_queue, batch_size=batch_size, num_threads=8, capacity=128)
	return image_batch, label_batch,file_xu_batch

def GetData_all(train_data,train_data_label,xu_protoset,itera,order,nb_cl):
	traindata_index = order[0:(itera+1)*nb_cl]
	images =[]
	label =[]
	file_xu=[]
	for i in traindata_index:
		index = np.where(train_data_label[0:50000] == i)
		images.append(train_data[index])
		label.append(train_data_label[index])
		file_xu.append(index)
	#添加保留集的数据信息
	for i in range(100):
		# 添加图像信息
		images.append(train_data[xu_protoset[i]])
		# 添加图像标签
		label.append(train_data_label[xu_protoset[i]])
		file_xu.append(xu_protoset[i])

	file_xu = np.concatenate(file_xu)
	file_xu = np.concatenate(file_xu)
	images = np.concatenate(images)
	label = np.concatenate(label)
	return images,label,file_xu

def Prepare_train_data_batch_all(train_data,train_data_label,xu_protoset,itera,order,nb_cl,batch_size=128):
	images,label,file_protoset = GetData_all(train_data,train_data_label,xu_protoset,itera,order,nb_cl)
	images = tf.cast(images, tf.float32)
	label = tf.cast(label, tf.int32)
	file_protoset = tf.cast(file_protoset, tf.int32)
	# 从tensor列表中按顺序或随机抽取一个tensor
	input_queue = tf.train.slice_input_producer([images, label,file_protoset], shuffle=True)
	image_batch, label_batch,file_protoset_batch = tf.train.batch(input_queue, batch_size=batch_size, num_threads=8, capacity=128)
	return image_batch, label_batch,file_protoset_batch

#获得测试数据
def GetTestData(test_data,test_data_label,itera,order,nb_cl):
	traindata_index = order[0:(itera+1)*nb_cl]
	images =[]
	label =[]
	file_xu=[]
	for i in traindata_index:
		index = np.where(test_data_label[0:10000] == i)
		images.append(test_data[index])
		label.append(test_data_label[index])
		file_xu.append(index)#图片在矩阵中的序号 可以视作文件名

	file_xu = np.concatenate(file_xu)
	file_xu = np.concatenate(file_xu)
	images = np.concatenate(images)
	label = np.concatenate(label)
	return images,label,file_xu

def Prepare_test_data_batch(test_data,test_data_label,itera,order,nb_cl,batch_size=128):
	images,label,file_protoset = GetTestData(test_data,test_data_label,itera,order,nb_cl)
	images = tf.cast(images, tf.float32)
	label = tf.cast(label, tf.int32)
	file_protoset = tf.cast(file_protoset, tf.int32)
	# 从tensor列表中按顺序或随机抽取一个tensor
	input_queue = tf.train.slice_input_producer([images, label,file_protoset], shuffle=True)
	image_batch, label_batch,file_protoset_batch = tf.train.batch(input_queue, batch_size=batch_size, num_threads=8, capacity=128)
	return image_batch, label_batch,file_protoset_batch
'''
#函数名 : reading_data_and_preparing_network
#作者:magic
#日期:2019.6.6
#作用:获得数据样本的模型输出特征(未执行 尚在参数阶段 执行sess.run 后获得)
#参数:数据样本,对应样本的标签,batch_size
#返回:返回网络参数和样本对应的特征。
'''

def reading_data_and_preparing_network(option,train_data,train_data_label,xu_protoset, itera, batch_size, order,nb_cl, save_path):
	if option == 'train':
		image_batch, label_batch,file_xu_batch = Prepare_train_data_batch(train_data,train_data_label,xu_protoset,itera,order,nb_cl,batch_size=128)
	elif option == 'test':
		image_batch, label_batch, file_xu_batch = Prepare_test_data_batch(train_data,train_data_label, itera,order, nb_cl, batch_size=128)
	label_batch_one_hot = tf.one_hot(label_batch, 100)
	### Network and loss function
	with tf.variable_scope('ResNet34'):
		with tf.device('/gpu:0'):
			scores = utils_cifar.ResNet34(image_batch, phase='test')
			graph = tf.get_default_graph()
			op_feature_map = graph.get_operation_by_name('ResNet34/pool_last/avg').outputs[0]

	loss_class = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=label_batch_one_hot, logits=scores))

	### Initilization
	#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>注意模型保存路径<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
	params = dict(cPickle.load(open(save_path + 'model-iteration' + str(nb_cl) + '-%i.pickle' % itera,'rb')))
	inits = utils_cifar.get_weight_initializer(params)

	return inits, scores, label_batch, loss_class, file_xu_batch,op_feature_map

'''
#函数名 : reading_data_and_preparing_network
#作者:magic
#日期:2019.6.6
#作用:获得数据样本的模型输出特征
#参数:数据样本,对应样本的标签,batch_size
#返回:返回网络参数和样本对应的特征。
'''
def load_class_in_feature_space(nb_cl, batch_size, scores, label_batch, loss_class, file_xu_batch,op_feature_map, sess,file_num):
	label_dico = []
	Dtot = []
	processed_file = []
	for i in range(int(np.ceil(file_num / batch_size) + 1)):#执行的次数
		sc, l, loss,file_tmp, feat_map_tmp = sess.run(
			[scores, label_batch, loss_class, file_xu_batch,op_feature_map])#样本得分 一个batch的样本标签 交叉熵损失 特征输出
		processed_file.extend(file_tmp)
		label_dico.extend(l)
		mapped_prototypes = feat_map_tmp[:, 0, 0, :]
		Dtot.append((mapped_prototypes.T) / np.linalg.norm(mapped_prototypes.T, axis=0))#np.linalg.norm 二范数 归一化

	Dtot = np.concatenate(Dtot, axis=1)
	label_dico = np.array(label_dico)
	processed_file = np.array(processed_file)
	return Dtot, label_dico, processed_file