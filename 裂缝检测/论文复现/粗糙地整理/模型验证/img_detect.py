import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import cv2
from torch.autograd import Variable



import math
import torch.nn as nn

class Crack(nn.Module):
    def __init__(self, Crack_cfg):
        super(Crack, self).__init__()
        self.features = self._make_layers(Crack_cfg)
        self.linear1 = nn.Linear(32*6*6,64)
        self.linear2 = nn.Linear(64,64)
        self.linear3 = nn.Linear(64,2)


    def forward(self, x):
        out = self.features(x)
#         print(out.size())
        out = out.view(out.size(0), -1)
        out = self.linear1(out)
        out = self.linear2(out)
        out = self.linear3(out)
        return out

    def _make_layers(self, cfg):
        """
        cfg: a list define layers this layer contains
            'M': MaxPool, number: Conv2d(out_channels=number) -> BN -> ReLU
        """
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
            
#         layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)
    #输出图片
def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated
# 识别图片
def predict_img(inputs):
    with torch.no_grad():
        inputs = inputs.to(device)
        outputs = model_t(inputs)
        imgNum = inputs.size()[0]
        _,preds = torch.max(outputs,1)
        std = []
        print(preds)
        for j in range(inputs.size()[0]):
            if preds[j] == 0:
                std.append(j)
        return std
def ReadImg(path,index):
	# 读取图片
	# DIRECTORY = "E:\\1裂缝检测\\testdata"
	# DIRECTORY = "E:\\1裂缝检测\\D\\CD"
	files = os.listdir(path)
	file_paths=[]#构造一个存放图片的列表数据结构
	imgs = []
	for file in files:
	    file_path= path +"\\" + file
	    file_paths.append(file_path)
	#     here is a loading of local image,it will take a long time
	#     img = io.imread(file_path)
	#     imgs.append(img)
	plt.figure()
	temp = io.imread(file_paths[index])
	# plt.imshow(temp)

	#plt.pause(0.001)  # pause a bit so that plots are updated
	return temp


def classify(img,imgNum):
	# 切分为imgNum^2份

	# 分割图片
	img_roi = []
	cord = []
	for i in range(14,img.shape[0]-13,27):  # [1]480*360==15*11---height
	    for j in range(14,img.shape[1]-13,27):  # [2]column-----------width
	        img_roi.append(img[(i-14):((i + 13) ), (j -14):((j + 13))])
	        cord.append([i,j])
	    print(i)

	# 转换图片并开始识别
	t = img_roi[0]
	t = to_pil(t)
	t = test_trainsforms(t).float()
	t = t.unsqueeze_(0)
	# print(t.size())
	for i in range(len(img_roi)):
	    if i != 0 :
	        te = img_roi[i]
	        te = to_pil(te)
	        te = test_trainsforms(te).float()
	        te = te.unsqueeze_(0)
	        t = torch.cat((t,te),0)
	        # print(t.size())
	st = predict_img(t)
	# print(st)
	# 对目标进行定位
	loc = []
	for i in st:
	    loc.append(cord[i])
	# print(loc)
	# 在图片上标记目标
	for i in loc:
	    cv2.rectangle(img, (i[0]-14,i[1]-14), (i[0]+13,i[1]+13), (255,0,0), 3)
	# cv2.rectangle(img, 左上角, 右下角, （r，g，b）, 粗细（1，2，3，，）)
	# 返回已画好的图片
	return img 
	# 显示
	# plt.imshow(temp)
	# plt.show()


Crack_cfg = {
    'Crack11':[16,16,'M',32,32,'M']
}
  
model_t = Crack(Crack_cfg['Crack11']);




to_pil = transforms.ToPILImage()
classname = ['crack','no_crack']
device = 'cpu'


# 此处是python2.7的模型用于python3出现问题的解决
from functools import partial
import pickle
import torch


model = torch.load('./crack0.9373REC_0.9331.pt')
model_t.load_state_dict(model)

model_t.eval()

device = "cpu"


# 定义变换
test_trainsforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
    ])

if __name__ == '__main__':
	imgNum = 6 # 图片分块数目：imgNum * imgNum
	# path = "E:\\1裂缝检测\\testdata" # 文件路径
	path  = "I:\\1裂缝检测\\CrackForest-dataset\\image\\"
	st = './img/'


	files = os.listdir(path)
	file_paths = [file_name for file_name in files if file_name.endswith(".jpg")]

	print(file_paths)

	for i in range(1,len(file_paths)):
		img = io.imread(path + file_paths[i])
		img = classify(img,imgNum) # 将图片分成imgNum*imgNum份进行识别
		# plt.imshow(img)

		img_name = st + str(i) + '.jpg'
		print(img_name)
		cv2.imwrite(img_name,img)
		print(i)