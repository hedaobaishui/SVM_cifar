import  numpy as  np
import svmtree
try:
	import cPickle
except:
	import pickle as cPickle

files_protoset     = []

for i in range(100):
    files_protoset.append([])
# Select the order for the class learning
order = np.load('./order.npy', encoding='latin1')
itera =0
nb_protos = 100
alph = 5
nu_cl_itera = 10
num_cl = 10
cl_feature_svm = cPickle.load(open('cl_feature_svm.pickle', 'rb'))
label_for_svm  = cPickle.load(open('label_for_svm.pickle', 'rb'))
nu_cl_for_svm  = cPickle.load(open('nu_cl_for_svm.pickle', 'rb'))
cl_list        = cPickle.load(open('cl_list.pickle', 'rb'))
pic_name       = cPickle.load(open('pic_name.pickle', 'rb'))
protoset_tmp = []
for i in range(10):
    protoset_tmp.append([])

svmtree.svm_recursion_fixed_nu_proto(cl_feature_svm, label_for_svm, nu_cl_for_svm, cl_list, num_cl,
                                     protoset_tmp, itera, nb_protos, alph, nu_cl_itera, pic_name)
for i in range(num_cl):
    class_index = order[itera*num_cl+i]
    files_protoset[class_index] = protoset_tmp[i]
print(files_protoset)