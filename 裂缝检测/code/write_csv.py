import pandas as pd
import os
import numpy as np



path  = "./D/CD/"
files = os.listdir(path)
file_paths=[]#构造一个存放图片的列表数据结构
a = []
for file in files:
    file_path= path + file
    file_paths.append(file_path)
    a.append(0)

path  = "./D/UD/"
files = os.listdir(path)
for file in files:
    file_path= path + file
    file_paths.append(file_path)
    a.append(1)
# print(arr[0:len(file_paths)])
#字典中的key值即为csv中列名
dataframe = pd.DataFrame({'img_dir':file_paths,'value':a})

#将DataFrame存储为csv,index表示是否显示行名，default=True
dataframe.to_csv("test.csv",index=False,sep=',')
