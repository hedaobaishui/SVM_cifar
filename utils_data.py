import numpy as np
import os
import scipy.io
import sys
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
	file=open(Cifar_file,'rb')
	dict=cPickle.load(file)
	file.close()
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
	xs.append(d['fine_labels'])
	
	#求解图像均值
	x = np.concatenate(xs)/np.float32(255)#将xs连接起来
	y = np.concatenate(ys)
	
	x = np.dstack((x[:,:1024],x[:,1024:2048]),x[:,2048:])#原数据1*1024*3（RGB）
	x = x.reshape((x.reshape[0],32,32,3)).transpose(0,3,1,2)#后三个位置的坐标转换 变成 5000*3*32*32
	
	#减去每个像素的均值
	pixel_mean = np.mean(x[0:5000],axis=0)
	x -=pixel_mean
	
	#创建训练集
	train_sample_cl=500-nb_per_cl_val
	X_train = np.zeros((train_sample_cl*100,3,32,32))
	Y_train = np.zeros(train_sample_cl*100)
	X_vaild = np.zeros((nb_per_cl_val*100,3,32,32))
	Y_vaild = np.zeros(nb_per_cl_val*100)
	for i in range(100):#一次性将所有样本都分配好了
		index_y=np.where(y[0:50000]==i)
		np.random.shuffle(index_y)
		X_train[i*train_sample_cl:(i+1)*train_sample_cl]=x[index_y[0:train_sample_cl],:,:,:]
		Y_train[i*train_sample_cl:(i+1)*train_sample_cl]=y[index_y[0:train_sample_cl]]
		X_vaild[i*nb_per_cl_val:(i+1)*nb_per_cl_val]=x[index_y[train_sample_cl:500],:,:,:]
		Y_vaild[i*nb_per_cl_val:(i+1)*nb_per_cl_val]=y[index_y[train_sample_cl:500]]
		
	X_test = x[50000:,:,:,:]
	Y_test = y[50000:]
	
	#这里数据都有了将数据转化为tensor
	#需要将数据转化为tensorflow的数据 还要考虑batch_size。
	
	return dict(
	X_train = X_train,
	Y_train = Y_train,
	X_test = X_test,
	Y_test = Y_test,
	)

'''
#函数名 : prepare_train_data
#作者:magic
#日期:2019.4.19
#作用:准备可以用来训练模型的数据
#参数:数据集,对应样本的标签,batch_size
#返回:返回准备好的数据集和对应的标签
'''
def prepare_train_data_batch(train_data,train_data_label,batch_size=128):
    iter=0
    data_batch=list()
    data_label=list()
    _data_batch=train_data['X_train'][batch_size*iter:batch_size*(iter+1)]
    _data_label=train_data_label['Y_train'][batch_size*iter:batch_size*(iter+1)]
    for x,y in _data_batch,_data_label:
        data_batch.append(x)
        hot=np.zeros(100)
        hot[int(y)]=1
        data_label.append(hot)
    return data_batch,data_label