import cPickle
import numpy as np
import tensorflow as tf

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
	X_train = tf.convert_to_tensor(X_train,dtype=tf.float32),
	Y_train = tf.convert_to_tensor(Y_train,dtype=tf.int32),
	X_vaild = tf.convert_to_tensor(X_vaild,dtype=tf.float32),
	Y_vaild = tf.convert_to_tensor(Y_vaild,dtype=tf.int32),
	X_test = tf.convert_to_tensor(X_test,dtype=tf.float32),
	Y_test = tf.convert_to_tensor(Y_test,dtype=tf.int32),
	)
