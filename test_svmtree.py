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
files_protoset = []
for i in range(100):
    files_protoset.append([])

svmtree.svm_recursion_fixed_nu_proto(cl_feature_svm, label_for_svm, nu_cl_for_svm, cl_list, num_cl,
                                     files_protoset,  nb_protos, alph, pic_name)
print(files_protoset)