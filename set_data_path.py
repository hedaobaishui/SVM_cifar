import  platform

'''
#函数名 : get_data_path()
#作者:magic
#日期:2019.6.8
#作用:获得在Linux 和 Windows 下的训练数据集 测试集 保存模型的路径
#参数:无
#返回:训练数据集 测试集 保存模型的路径
'''
def get_data_path():
    if platform.system() == 'Windows':
        Cifar_train_file = 'F:/Dataset/ILSVRC2012/cifar-100-python/train'
        # 需要修改
        Cifar_test_file = 'F:/Dataset/ILSVRC2012/cifar-100-python/test'  # 需要修改
        save_path = './model/'
    elif platform.system() == 'Linux':
        Cifar_train_file = '/home/magic/project/virtualenv/TF/cifar-100-python/train'
        # 需要修改
        Cifar_test_file = '/home/magic/project/virtualenv/TF/cifar-100-python/test'  # 需要修改
        save_path = './model/'
    return Cifar_train_file, Cifar_test_file, save_path


if __name__== '__main__':
    Cifar_train_file, Cifar_test_file, save_path = get_data_path()
    print(Cifar_train_file, Cifar_test_file, save_path)