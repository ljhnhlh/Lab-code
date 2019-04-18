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
import csv

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
std=[[0,0],[0,0]]
# 识别图片
def predict_img(model,inputs,index):
	with torch.no_grad():
		inputs = inputs.to(device)
		outputs = model(inputs)
		imgNum = inputs.size()[0]
		_,preds = torch.max(outputs,1)  
		# print(preds)
		# print(inputs.size()[0])
		for j in range(inputs.size()[0]):
			if preds[j] == 0 and values[j+index] == '0':
				std[0][0]+=1
			elif preds[j] == 0 and values[j+index] == '1':
				std[0][1]+=1
			elif preds[j] == 1 and values[j+index] == '0':
				std[1][0]+=1
			elif preds[j] == 1 and values[j+index] == '1':
				std[1][1]+=1


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


def classify(img):
	
	# 转换图片并开始识别
	t = img[0]
	t = to_pil(t)
	t = test_trainsforms(t).float()
	t = t.unsqueeze_(0)
	# print(t.size())
	print('start')
	for i in range(len(img_roi)):
		te = img_roi[i]
		te = to_pil(te)
		te = test_trainsforms(te).float()
		te = te.unsqueeze_(0)
		# t为四元组，每一维都是一张照片
		t = torch.cat((t,te),0)
		# print(t.size())
	print('end')
	# predict_img(t)
	return t;


#加载模型
from functools import partial
import pickle
import torch
pickle.load = partial(pickle.load, encoding="latin1")
pickle.Unpickler = partial(pickle.Unpickler, encoding="latin1")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = torch.load('./res.pt',map_location=lambda storage, loc: storage, pickle_module=pickle)
model = model.to(device)
model.eval()

to_pil = transforms.ToPILImage()
classname = ['crack','no_crack']

# model.eval()
# 定义变换
test_trainsforms = transforms.Compose([
		transforms.Resize(256),
		transforms.CenterCrop(224),
		transforms.ToTensor(),
		transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
	])	
filepaths = []
values = []
if __name__ == '__main__':
	# 读取csv至字典
	csvFile = open("test.csv", "r")
	reader = csv.reader(csvFile)
	# 建立空字典
	for item in reader:
	    # 忽略第一行
	    if reader.line_num == 1:
	        continue
	    filepaths.append(item[0])
	    values.append(item[1])
	csvFile.close()

	
	img_roi=[]
	for i in range(len(filepaths)):
		img = io.imread(filepaths[i])
		img_roi.append(img);
		if(i % 100 == 0):
			t = classify(img_roi) 
			predict_img(model,t,i-100)
			img_roi = []

	print(std)
	print("res:精确率：{}".format(std[0][0]/(std[0][0]+std[0][1])))
	print("res:召回率：{}".format(std[0][0]/(std[0][0]+std[1][0])))
	# model = torch.load('./model1.pt',map_location=lambda storage, loc: storage, pickle_module=pickle)
	# model = model.to(device)
	# model.eval()
	# std = predict_img(model,t)
	# print(std)
	# print("model1:精确率：{}".format(std[0][0]/(std[0][0]+std[0][1])))
	# print("model1:召回率：{}".format(std[0][0]/(std[0][0]+std[1][0])))

	# # model = torch.load('./model2.pt',map_location=lambda storage, loc: storage, pickle_module=pickle)
	# # model = model.to(device)
	# # model.eval()
	# # std = predict_img(model,t)
	# # print("model2:精确率：{}".format(std[0][0]/(std[0][0]+std[0][1])))
	# # print("model2:召回率：{}".format(std[0][0]/(std[0][0]+std[1][0])))
	# # print(std)

	# model = torch.load('./res.pt', map_location=lambda storage, loc: storage, pickle_module=pickle)
	# model = model.to(device)
	# model.eval()
	# std = predict_img(model,t)
	# print("res:精确率：{}".format(std[0][0]/(std[0][0]+std[0][1])))
	# print("res:召回率：{}".format(std[0][0]/(std[0][0]+std[1][0])))
	# print(std)

	# model = torch.load('./resnet50.pt',map_location=lambda storage, loc: storage, pickle_module=pickle)
	# model = model.to(device)
	# model.eval()
	# std = predict_img(model,t)
	# print("resnet50:精确率：{}".format(std[0][0]/(std[0][0]+std[0][1])))
	# print("resnet50:召回率：{}".format(std[0][0]/(std[0][0]+std[1][0])))
	# print(std)

	# model = torch.load('./resNet152.pt',map_location=lambda storage, loc: storage, pickle_module=pickle)
	# model = model.to(device)
	# model.eval()
	# std = predict_img(model,t)
	# print("resNet152:精确率：{}".format(std[0][0]/(std[0][0]+std[0][1])))
	# print("resNet152:召回率：{}".format(std[0][0]/(std[0][0]+std[1][0])))
	# print(std)

	# model = torch.load('./best.pt',map_location=lambda storage, loc: storage, pickle_module=pickle)
	# model = model.to(device)
	# model.eval()
	# std = predict_img(model,t)
	# print("best:精确率：{}".format(std[0][0]/(std[0][0]+std[0][1])))
	# print("best:召回率：{}".format(std[0][0]/(std[0][0]+std[1][0])))
	# print(std)