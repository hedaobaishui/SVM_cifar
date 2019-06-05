import numpy as np
import tensorflow as tf
import os
import scipy.io
import sys
from scipy.spatial.distance import cdist

try:
	import cPickle
except:
	import pickle as cPickle


def relu(x,name,alpha):
	if alpha>0:
		return tf.maxinum(alpha*x,x,name=name)
	else:
		return tf.nn.relu(x,name=name)

	
def get_variable(name,shape,dtype,initializer,trainable=True,regularizer=None):
	with tf.device('/cpu:0'):
		var = tf.get_variable(name,shape=shape,dtype=dtype,
								initializer=initializer,regularizer=regularizer,trainable=trainable,
								collections=[tf.GraphKeys.WEIGHTS,tf.GraphKeys.GLOBAL_VARIABLES])
	return var

def conv(inp,name,size,out_channels,strides=[1,1,1,1],
				dilation=None,padding='SAME',apply_relu=True,alpha=0.0,bias=True,
				initializer=tf.contrib.layers.xavier_initializer_conv2d()):
	batch_size = inp.get_shape().as_list()[0]#batch_size
	res1 = inp.get_shape().as_list()[1]#width
	res2 = inp.get_shape().as_list()[1]#height
	in_channels = inp.get_shape().as_list()[3]#channel
	
	with tf.variable_scope(name):
		#size 是卷积核的大小
		W = get_variable("W",shape=[size,size,in_channels,out_channels],dtype=tf.float32,
								initializer=initializer,regularizer=tf.nn.l2_loss)
		b = get_variable("b",shape=[1,1,1,out_channels],dtype=tf.float32,
								initializer=tf.zeros_initializer(),trainable=bias)
		
		if dilation:# 膨胀
			assert(strides==[1,1,1,1])
			out = tf.add(tf.nn.atrous_conv2d(inp,W,rate=dilation,padding=padding),b,name='convolution')
			out.set_shape([batch_size,res1,res2,out_channels])
		else:
			out = tf.add(tf.nn.conv2d(inp,W,strides,padding=padding),b,name='convolution')
		
		if apply_relu:
			out = relu(out,'relu',alpha)
	return out
								
def softmax(target,axis,name=None):
	max_axis = tf.reduce_max(target,axis,keep_dims=True)#目标某个维度上的最大值
	target_exp = tf.exp(target-max_axis)#减去最大值后的exp
	normalize = tf.reduce_sum(target_exp,axis,keep_dims=True)
	softmax = target_exp / normalize
	return softmax

def batch_norm(inp,name,phase,decay=0.9):
	channels = inp.get_shape().as_list()[3]
	with tf.variable_scope(name):
		moving_mean = get_variable("mean",shape=[channels],dtype=tf.float32,initializer=tf.constant_initializer(0.0),trainable=False)
		moving_variance = get_variable("var",shape=[channels],dtype=tf.float32,initializer=tf.constant_initializer(1.0),trainable=False)
		
		offset = get_variable("offset",shape=[channels],dtype=tf.float32,initializer=tf.constant_initializer(0.0))
		scale = get_variable("scale",shape=[channels],dtype=tf.float32,initializer=tf.constant_initializer(1.0),regularizer=tf.nn.l2_loss)
		
		mean,variance = tf.nn.moments(inp,axes=[0,1,2],shift=moving_mean)
		
		mean_op=moving_mean.assign(decay*moving_mean+(1-decay)*mean)
		var_op = moving_variance.assign(decay*moving_variance+(1-decay)*variance)
		
		assert(phase in ['train','test'])
		if phase == 'train':
			with tf.control_dependencies([mean_op,var_op]):
				return tf.nn.batch_normalization(inp,mean,variance,offset,scale,0.01,name='norm')
		else:
			return tf.nn.batch_normalization(inp,moving_mean,moving_variance,offset,scale,0.01,name='norm')
	
def pool(inp,name,kind,size,stride,padding='SAME'):
	assert kind in ['max','avg']
	
	strides=[1,stride,stride,1]
	sizes = [1,size,size,1]
	
	with tf.variable_scope(name):
		if kind == 'max':
			out = tf.nn.max_pool(inp,sizes,strides=strides,padding=padding,name=kind)
		else:
			out = tf.nn.avg_pool(inp,sizes,strides=strides,padding=padding,name=kind)
	return out

def residual_block(inp,phase,alpha=0.0,nom='a',increase_dim=False,last=False):
	input_num_filters = inp.get_shape().as_list()[3]
	
	if increase_dim :
		first_stride = [1,1,2,2]
		out_num_filters = input_num_filters*2 #卷积核数量扩大一倍
	else:
		first_stride = [1,1,1,1]
		out_num_filters = input_num_filters
	
	layer = conv(inp,'resconv1'+nom,size=3,strides=first_stride,out_channels=out_num_filters,alpha=alpha,padding='SAME')
	layer = batch_norm(layer,'batch_norm_resconv1'+nom,phase=phase)
	layer = conv(layer,'resconv12'+nom,size=3,strides=[1,1,1,1],out_channels=out_num_filters,apply_relu=False,alpha=alpha,padding='SAME')
	
	if increase_dim :
		projection = conv(inp,'projconv'+nom,size=1,strides=[1,1,2,2],out_channels=out_num_filters,alpha=alpha,apply_relu=False,padding='SAME',bias=False)
		projection = batch_norm(projection,'batch_norm_projconv'+nom,phase=phase)
		if last :
			block = layer +projection
		else:
			block = layer+ projection
			block = tf.nn.relu(block,name='relu')
	else:
		if last:
			block = layer +inp
		else:
			block = layer +inp
			block = tf.nn.relu(block,name='relu')
	return block	

#multiple teacher and one student
#itera:第几次增量学习
#inp:输入模型
#固定旧block的参数。
#应该是加载旧模型，然后在就模型的基础上进行添加分支。
#先准备网络，将先前训练好的网络参数复制到新网络中。
#然后添加分支。固定分支网络不可训练。	


#先准备一个共享网络块
#34层的残差网络15个BLOCK
#共享3个block
def ResNet34(inp, phase, num_outputs=100,itera=0,alpha=0.0):#phase test or train
    # First conv
    # first layer, output is 16 x 32 x 32
	layer = conv(inp,"conv1",size=3,strides=[1, 1, 1, 1], out_channels=16, alpha=alpha, padding='SAME')
	layer = batch_norm(layer, 'batch_norm_1', phase=phase)
	#layer = pool(layer, 'pool1', 'max', size=3, stride=2)
    
    # first stack of residual blocks, output is 32 x 32 x16
	for letter in 'abcde':
		layer = residual_block(layer, phase, alpha=0.0,nom=letter)
    
    # second stack of residual blocks, output is 16 x 16 x32
	layer = residual_block(layer, phase, alpha=0.0,nom='f',increase_dim=True)
	for letter in 'ghij':
		layer = residual_block(layer, phase, alpha=0.0,nom=letter)
    
	# Third stack of residual blocks,output is 64 x 8 x 8
	layer = residual_block(layer, phase, alpha=0.0,nom='k',increase_dim=True)
	for letter in 'lmn':
		layer = residual_block(layer, phase, alpha=0.0,nom=letter)


	layer = residual_block(layer, phase, alpha=0.0, nom='o', increase_dim=True)
	layer = residual_block(layer, phase, alpha=0.0, nom='p', last=True)
# 后面添加全连接层
	layer = pool(layer, 'pool_last', 'avg', size=4, stride=1, padding='VALID')
	layer = conv(layer, name='fc', size=1, out_channels=num_outputs, padding='VALID', apply_relu=False, alpha=alpha)[:, :,0, 0]
	return layer


def prepareNetwork(gpu,image_batch):
	scores = []
	with tf.variable_scope('ResNet34'):
		with tf.device('/gpu:' + gpu):
			score = ResNet34(image_batch, phase='train')
			scores.append(score)

		scope = tf.get_variable_scope()
		scope.reuse_variables()

	# First score and initialization
	variables_graph = tf.get_collection(tf.GraphKeys.WEIGHTS, scope='ResNet34')
	scores_stored = []
	with tf.variable_scope('store_ResNet34'):
		with tf.device('/gpu:' + gpu):
			score = ResNet34(image_batch, phase='test')
			scores_stored.append(score)

		scope = tf.get_variable_scope()
		scope.reuse_variables()

	variables_graph2 = tf.get_collection(tf.GraphKeys.WEIGHTS, scope='store_ResNet34')

	return variables_graph, variables_graph2, scores, scores_stored


def get_weight_initializer(params):
	initializer = []
	scope = tf.get_variable_scope()
	scope.reuse_variables()
	for layer, value in params.items():
		op = tf.get_variable('%s' % layer).assign(value)
		initializer.append(op)
	return initializer



def save_model(name, scope, sess):
	variables = tf.get_collection(tf.GraphKeys.WEIGHTS, scope=scope)
	d = [(v.name.split(':')[0], sess.run(v)) for v in variables]
	cPickle.dump(d, open(name, 'wb'))

def accuracy_measure(X_valid, Y_valid, class_means, val_fn, top1_acc_list, iteration, iteration_total, type_data):
	stat_hb1 = []
	stat_icarl = []
	stat_ncm = []

	for batch in iterate_minibatches(X_valid, Y_valid, min(500, len(X_valid)), shuffle=False):
		inputs, targets_prep = batch
		targets = np.zeros((inputs.shape[0], 100), np.float32)
		targets[range(len(targets_prep)), targets_prep.astype('int32')] = 1.
		err, pred, pred_inter = val_fn(inputs, targets)
		pred_inter = (pred_inter.T / np.linalg.norm(pred_inter.T, axis=0)).T

		# Compute score for iCaRL
		sqd = cdist(class_means[:, :, 0].T, pred_inter, 'sqeuclidean')
		score_icarl = (-sqd).T
		# Compute score for NCM
		sqd = cdist(class_means[:, :, 1].T, pred_inter, 'sqeuclidean')
		score_ncm = (-sqd).T

		# Compute the accuracy over the batch
		stat_hb1 += ([ll in best for ll, best in zip(targets_prep.astype('int32'), np.argsort(pred, axis=1)[:, -1:])])
		stat_icarl += (
		[ll in best for ll, best in zip(targets_prep.astype('int32'), np.argsort(score_icarl, axis=1)[:, -1:])])
		stat_ncm += (
		[ll in best for ll, best in zip(targets_prep.astype('int32'), np.argsort(score_ncm, axis=1)[:, -1:])])

	print("Final results on " + type_data + " classes:")
	print("  top 1 accuracy iCaRL          :\t\t{:.2f} %".format(np.average(stat_icarl) * 100))
	print("  top 1 accuracy Hybrid 1       :\t\t{:.2f} %".format(np.average(stat_hb1) * 100))
	print("  top 1 accuracy NCM            :\t\t{:.2f} %".format(np.average(stat_ncm) * 100))

	top1_acc_list[iteration, 0, iteration_total] = np.average(stat_icarl) * 100
	top1_acc_list[iteration, 1, iteration_total] = np.average(stat_hb1) * 100
	top1_acc_list[iteration, 2, iteration_total] = np.average(stat_ncm) * 100

	return top1_acc_list
								
								
								
								
								
								
								
								
								
								
								
								
								
								
								
								
								
								
								
								
								
								
								
								
								
								
								
								
								
								
								
								
								
								
								
								
								
								
								
								
								
								
								
								
								
								
								
								
								
								
								
								
								
								
								
								
								
								
								
								
								
								
								
								
								
								
								
								
								
								
								
								
								
								
								
								
								
								
								
								
								
								
								
								
