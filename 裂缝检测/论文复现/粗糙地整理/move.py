import os 
import shutil as sh 

positivePath = "I:/1裂缝检测/CrackForest-dataset/trian/positive/"
negativePath = "I:\\1裂缝检测\\CrackForest-dataset\\trian\\negative4\\"

train_po_path = "I:\\1裂缝检测\\CrackForest-dataset\\trian\\train2\\train\\crack\\"
train_ne_path = "I:\\1裂缝检测\\CrackForest-dataset\\trian\\train2\\train\\no_crack\\"

val_po_path = "I:/1裂缝检测/CrackForest-dataset/trian/train2/val/crack/"
val_ne_path = "I:/1裂缝检测/CrackForest-dataset/trian/train2/val/no_crack/"


po_file_list = os.listdir(positivePath)
ne_file_list = os.listdir(negativePath)
# print(po_file_list)

for j in range(len(po_file_list)):
    if(j < 1000):
        sh.move(positivePath+po_file_list[j],train_po_path+po_file_list[j])
    else:
        sh.move(positivePath+po_file_list[j],val_po_path+po_file_list[j])
    if(j % 1000 == 0):
        print(j)
    if(j > 2000):
        break;

for j in range(len(ne_file_list)):
    if(j < 2000):
        sh.move(negativePath+ne_file_list[j],train_ne_path+ne_file_list[j])
    else:
        sh.move(negativePath+ne_file_list[j],val_ne_path+ne_file_list[j])
    if(j > 4000):
        break
    if(j % 1000 == 0):
        print(j)

    