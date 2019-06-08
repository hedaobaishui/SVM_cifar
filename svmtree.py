#实现SVMT
import numpy as np
import cv2 as cv
import string
import operator
#svm参数配置
def svm_config():
    svm = cv.ml_SVM.create()
    svm.setCoef0(0.0)
    #svm.setKernel(cv.ml.SVM_LINEAR)
    # svm.setDegree(3)
    #svm.setGamma(0)
    svm.setKernel(cv.ml.SVM_POLY)
    svm.setGamma(1)
    svm.setCoef0(0)
    svm.setDegree(1.0)
    criteria = (cv.TERM_CRITERIA_MAX_ITER + cv.TERM_CRITERIA_EPS, 1000, 1e-3)
    svm.setTermCriteria(criteria)
    #svm.setNu(0.5)
    #svm.setP(0.1)
    #svm.setC(0.01)
    #svm.setType(cv.ml.SVM_EPS_SVR)

    return svm

def svm_train(svm,data,labels):
    #特征以及样本啊！
    traindata=cv.ml_TrainData.create(data,cv.ml.ROW_SAMPLE,labels)
    svm.train(traindata)
def svm_save(svm,name):
    svm.save(name)

#svm加载参数
def svm_load(name):
    svm = cv.ml.SVM_load(name)
    return svm

#递归实现 SVMT
#file_feature: 从卷积层输出的特征 每一行是一个样本
#label：特征的标签。是类别标签[0,1,2,3,4.....1000]
#nu_cl_for_svm: 每一类样本的数量;[1250,1250,1250....]
#num_cl:当前需要分割的样本种类数量；5->2,3;2>1,3->2,1;2->1
#proto_file:存储提取的支持向量的序号
#itera:当前是第几次增量学习；一共200次增量学习
#nb_protos_cl:每个类别需要提取的proto 样本数量
#alph 每个分割节点选取的样本数量
#svm_recursion_fixed_nu_proto : 固定抽样样本的数量
#svm_recursion_vary_nu_proto : 抽样样本的数量在不断变化
#nu_cl_itera: 每次迭代的样本数量
# pic_name:样本的名字。也即图片名。
def svm_recursion_fixed_nu_proto(file_feature, label,nu_cl_for_svm, cl_list,num_cl,proto_file,itera,nb_protos_cl,alph,nu_cl_itera,pic_name):#一般 num_cl= 2,5,10,50

    #仅仅叶子节点
    if num_cl == 1:
        step=0
        #缺多少补多少样本
        #判断当前的类别
        cl_now=label[0]#当前类
        while len(proto_file[cl_now]) < nb_protos_cl and step<1.1*nb_protos_cl:
            if pic_name[step] not in proto_file[cl_now]:
                proto_file[cl_now].append(pic_name[step])
                step=step+1
        return proto_file

    #label:是样本来别标签;还需要加-1,1标签
    file_feature=np.float32(file_feature)
    label=np.int32(label)
    num_cl_left = int(np.ceil(num_cl/2))
    num_cl_right =int(np.floor(num_cl/2))
    #-1 和 1 的标签
    #类别样本的数量；
    #样本的数量 要不要改变nu_cl_for_svm的数量
    #获得当前原本的类别数
    #获得当前样本的数量
    num_file_left=np.sum(nu_cl_for_svm[cl]  for cl in cl_list[0:num_cl_left])
    num_file_right = np.sum(nu_cl_for_svm[cl] for cl in cl_list[num_cl_left:])
    label_01=np.array([-1]*num_file_left+[1]*num_file_right)
    #样本的标签
    #label 存在样本标签
    #label_true=np.array()
    #训练SVM
    #使用 np.where()
    #然后挑选样本
    svm=svm_config()
    label_01=np.int32(label_01)
    svm_train(svm,file_feature,label_01)
    kk=svm.getSupportVectors()#获得支持向量
    #根据KKT条件挑选样本挑选样本
    for svc in kk:
        loction=0
        loc_svc_list = []
        for lines in file_feature: #根据样本所属集合找到样本的类别
            if operator.eq(lines.tolist(),svc.tolist()):#属于哪一类的特征
                loc_svc_list.append(loction)
            loction+=1
        #loc_svc 可能是list
        for loc_svc in loc_svc_list:
            svc_true_label=label[loc_svc]# 找到样本的类别
            #svc_true_label=itera*num_cl+num_cl
            #在SVC 周围选取一些样本5个；使用曼哈顿距离选取。
            #提取该类的样本：
            itera_label=svc_true_label-(itera*nu_cl_itera)
            #获取当前类的特征数据
            if itera_label>0:
                file_now = file_feature[np.sum(nu_cl_for_svm[0:itera_label]):np.sum(nu_cl_for_svm[0:itera_label+1]), :]
            else:#==0
                file_now=file_feature[0:nu_cl_for_svm[itera_label],:]
            #要在图片信息和序号之间做一个匹配；
            #可以建立一个字典；
            #通过特征选择样本。
            dis=[]
            for feat in file_now:
                dis.extend(np.linalg.norm(feat-svc,ord=1,keepdims=True))
            #dis=np.array(np.linalg.norm(feat-svc,ord=1,keepdims=True) for feat in file_now)#使用曼哈顿距离 取距离最小的样本
            label_sort=np.argsort(dis)#[2,4,3,1]返回距离原位置的标签返回[3,0,2,1]
            #取样本
            #应该直接将图片名称加入 feat 是特征: 还要找到特征对应的图片名称。
            prototmp=label_sort[0:alph]# 取5个样本
            #添加的是当前训练样本的序号：
            for proto in prototmp:
                if len(proto_file[itera * nu_cl_itera + itera_label]) < nb_protos_cl:
                    proto_file[itera*nu_cl_itera +itera_label].append(pic_name[proto+int(np.sum(nu_cl_for_svm[0:itera_label]))])#
                else:
                    break

    if num_cl==2:#两个
        step=0
        #缺多少补多少样本
        #判断当前的类别
        label_ave=np.sum(label)/(len(label))
        labe1_l_r=[int(np.floor(label_ave)),int(np.ceil(label_ave))]
        for leaf in labe1_l_r:#两个类别都需要填满样本
           while len(proto_file[leaf]) < nb_protos_cl and step<1.1*nb_protos_cl:
                 if pic_name[step+int(np.sum(nu_cl_for_svm[0:leaf]))] not in proto_file[leaf]:
                      proto_file[leaf].append(pic_name[step+int(np.sum(nu_cl_for_svm[0:leaf]))])
                 step=step+1
        return proto_file

    #更新样本需要处理的数据:
    file_feature_left=file_feature[0:num_file_left,:]
    file_feature_right=file_feature[num_file_left:,:]
    pic_name_left=pic_name[0:num_file_left]
    pic_name_right=pic_name[num_file_left:]
    label_left=label[0:num_file_left]
    label_right=label[num_file_left:]
    cl_list_left=cl_list[0:num_cl_left]
    cl_list_right=cl_list[num_cl_left:]

    svm_recursion_fixed_nu_proto(file_feature_left, label_left,nu_cl_for_svm,cl_list_left,num_cl_left,proto_file,itera,nb_protos_cl,alph,nu_cl_itera,pic_name_left)
    svm_recursion_fixed_nu_proto(file_feature_right, label_right,nu_cl_for_svm,cl_list_right,num_cl_right,proto_file,itera,nb_protos_cl,alph,nu_cl_itera,pic_name_right)

#没有实现
def svm_recursion_vary_nu_proto(file_feature, label,nu_cl_for_svm, num_cl,proto_file,itera,nb_protos_cl):#一般 num_cl= 2,5,10,50
    return proto_file

if __name__== '__main__':
    #test file->>>>>>>>
    feature_test=[[0,0],[1,1],[1,2],[1,3],[2,1],[3,2],[3,3],[4,4],[3,4],[4,3],[6,6],[5,6],[6,7]]
    pic_name=['a1','a2','a3','a4','a5','b1','b2','b3','b4','b5','c1','c2','c3']
    label_svm_test=[0,0,0,0,0,1,1,1,1,1,2,2,2]
    feature_test=np.array(feature_test)
    label_svm_test=np.array(label_svm_test)
    feature_test=np.float32(feature_test)
    label_svm_test=np.int32(label_svm_test)
    nu_cl_for_svm_test=np.array([5,5,3])
    # <<<<<<<<<-test file
    num_cl = 3
    proto_file=[]
    for _ in range(1 * 3):  # 100*10
        proto_file.append([])
    itera=0
    nb_protos_cl=2
    nu_cl_itera=3
    alph=1
    cl_list=[0,1,2]#
    #先要预处理
    cl_list=[int(cl-itera*num_cl) for cl in cl_list]
    svm_recursion_fixed_nu_proto(feature_test, label_svm_test, nu_cl_for_svm_test,cl_list, num_cl, proto_file, itera, nb_protos_cl,alph,nu_cl_itera,pic_name)
    print(proto_file)
