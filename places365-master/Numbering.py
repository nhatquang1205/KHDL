import os
import numpy as np
import re
import matplotlib.pyplot as plt
from numpy.lib.type_check import mintypecode
folder = 'photos'
count = 0
mydict = dict()
for foldername in os.listdir(folder): 
    dict_temp = dict([(foldername,0)])
    mydict.update(dict_temp)
    for filename in os.listdir(os.path.join(folder,foldername)):
        mydict[foldername]+=1
        count += 1
for key in mydict.keys():
    print('{:<40}              {:>3}'.format(key,mydict[key]))
print('Tổng số lượng mẫu : ',count)

i = 0
smalldict1 = dict()
smalldict2 = dict()
smalldict3 = dict()
for key in mydict.keys():
    dict_temp = dict([(key,mydict[key])])
    if i < 5: 
        smalldict1.update(dict_temp)
        i = i + 1
    elif i < 10:
        smalldict2.update(dict_temp)
        i = i + 1
    else:
        smalldict3.update(dict_temp)
plt.xlabel('Các địa danh')
plt.ylabel('Số lượng mẫu')
plt.bar(smalldict1.keys(),smalldict1.values())
plt.show()
plt.bar(smalldict2.keys(),smalldict2.values())
plt.show()
plt.bar(smalldict3.keys(),smalldict3.values(),align= 'edge')


plt.bar(mydict.keys(),mydict.values(),align='edge')
plt.xticks(rotation= 30)
plt.show()