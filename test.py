import numpy as np

label = np.array([1,2,3,4])
logit = np.array([[1,2,36,7],[13,4,5,2],[23,32,3,4],[12,23,43,6]])
var1=[logit1.var() for logit1 in logit]
print(var1)
logitnew = logit.T
print(logit)
logit_tmp =  np.argsort(logitnew.T, axis=1)[:, -1:]
print(logit_tmp)
label_zip = zip(label, logit_tmp)
ind_get = label[[ll in best for ll, best in label_zip]]
var2=[logit1.var() for logit1 in logitnew.T]
print(var2)