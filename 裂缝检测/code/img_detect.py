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
        outputs = model(inputs)
        imgNum = inputs.size()[0]
        _,preds = torch.max(outputs,1)
        std = []
        print(preds)
        for j in range(inputs.size()[0]):
            # ax = plt.subplot(imgNum//2,2,j+1)
            # ax.axis('off')
            # ax.set_title('predicted: {}'.format(classname[preds[j]]))
            if preds[j] == 0:
                std.append(j)
            # imshow(inputs.cpu().data[j])
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

	imw = img.shape[0]//(imgNum)
	imh = img.shape[1]//(imgNum)
	# 分割图片
	img_roi = []
	for i in range(imgNum):  # [1]480*360==15*11---height
	    for j in range(imgNum):  # [2]column-----------width
	        img_roi.append(img[(i * imw):((i + 1) * imw), (j * imh):((j + 1) * imh)])

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
	    x = i % imgNum
	    y = i // imgNum
	    loc.append([x,y])
	# print(loc)
	# 在图片上标记目标
	for i in loc:
	    cv2.rectangle(img, (i[0]*imw,i[1]*imh), (i[0]*imw+imw,i[1]*imh+imh), (255,0,0), 3)
	# cv2.rectangle(img, 左上角, 右下角, （r，g，b）, 粗细（1，2，3，，）)
	# 返回已画好的图片
	return img 
	# 显示
	# plt.imshow(temp)
	# plt.show()
to_pil = transforms.ToPILImage()
classname = ['crack','no_crack']
device = 'cpu'
# 加载模型
# model = torch.load('./res.pt',map_location='cpu')
# model = torch.load('./resNet50.pt')
# model = torch.load('./best.pt')

# 此处是python2.7的模型用于python3出现问题的解决
from functools import partial
import pickle
import torch
pickle.load = partial(pickle.load, encoding="latin1")
pickle.Unpickler = partial(pickle.Unpickler, encoding="latin1")

from torchvision import datasets, models, transforms
model = models.resnet18(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 2)
model.load_state_dict(torch.load('./modelprecise_0.280976.pt', map_location=lambda storage, loc: storage, pickle_module=pickle))

model.eval()
# 定义变换
test_trainsforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

if __name__ == '__main__':
	imgNum = 6 # 图片分块数目：imgNum * imgNum
	# path = "E:\\1裂缝检测\\testdata" # 文件路径
	path  = "I:\\1裂缝检测\\new\\D\\CD"
	# st = 'E:\\1裂缝检测\\testdata\\img2\\'


	files = os.listdir(path)
	file_paths=[]#构造一个存放图片的列表数据结构
	for file in files:
	    file_path= path +"\\" + file
	    file_paths.append(file_path)
	st = './img7/'
	for i in range(len(file_paths)):
		img = io.imread(file_paths[i])
		img = classify(img,imgNum) # 将图片分成imgNum*imgNum份进行识别
		plt.imshow(img)
		img_name = st + str(i) + '.jpg'
		print(img_name)
		cv2.imwrite(img_name,img)
		if(i>100):
			break;