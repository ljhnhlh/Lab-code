
# import numpy as np
# import cv2

# cap = cv2.VideoCapture(r'E:\\1裂缝检测\\testdata\\V90407-093024.mp4')
# # cap = cv2.VideoCapture('output.avi')

# # while(cap.isOpened()):
# #     ret, frame = cap.read()

# #     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# #     cv2.imshow('frame',gray)
# #     if cv2.waitKey(1) & 0xFF == ord('q'):
# #         break


# # Define the codec and create VideoWriter object
# # size = Size(capture.get(CV_CAP_PROP_FRAME_WIDTH), capture.get(CV_CAP_PROP_FRAME_HEIGHT))
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out = cv2.VideoWriter('./output.mp4', fourcc,20.0, size)

# while(cap.isOpened()):
#     ret, frame = cap.read()
#     if ret==True:
#         # write the flipped frame
#         out.write(frame)

#         cv2.imshow('frame',frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#     else:
#         break

# # Release everything if job is finished
# cap.release()
# out.release()
# cv2.destroyAllWindows()



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
import numpy as np
np.random.seed(1024)

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
        # print(preds)
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
	colors = [[255, 0, 0],[255, 97, 0],[255, 255, 0],[0, 255, 0],[0, 255, 255],[0, 0, 255],[160, 32, 240]]
	for i in loc:
		t = np.random.randint(0,7)
		cv2.rectangle(img, (i[0]*imw,i[1]*imh), (i[0]*imw+imw,i[1]*imh+imh),colors[t],10)
	    # cv2.rectangle(img, (i[0]*imw,i[1]*imh), (i[0]*imw+imw,i[1]*imh+imh), (255,0,0), 50)
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
model = torch.load('./resnet50.pt',map_location='cpu')
model.eval()
# 定义变换
test_trainsforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

# if __name__ == '__main__':
# 	imgNum = 6 # 图片分块数目：imgNum * imgNum
# 	index = 2 # 图片序号
# 	# path = "E:\\1裂缝检测\\testdata" # 文件路径
# 	path  = "C:\\Users\\林俊浩\\Desktop\\大三下\\人工神经网络\\todayWorks\\data2"
# 	img =  ReadImg(path,index)	# 图片
# 	img = classify(img,imgNum) # 将图片分成imgNum*imgNum份进行识别
# 	plt.imshow(img)
# 	plt.show()
# 	plt.pause(0.001)
#=====================================================================================


import cv2
 
imgNum = 6 
# cv2.VideoWriter_fourcc("I", "4", "2", "0")  .avi的未压缩的YUV颜色编码，文件较大
# cv2.VideoWriter_fourcc("P", "I", "M", "1")  .avi的MPEG-1编码类型
# cv2.VideoWriter_fourcc("X", "V", "I", "D")  .avi的MPEG-4编码类型
# cv2.VideoWriter_fourcc("T", "H", "E", "O")  .ogv的Ogg Vorbis
# cv2.VideoWriter_fourcc("F", "L", "V", "1")  .flv的flash视频
path = 'E:\\1裂缝检测\\testdata\\vedio'
files = os.listdir(path)
file_paths=[]
for file in files:
	file_path = path + "\\" + file
	file_paths.append(file_path)
t1 = cv2.getTickCount()  # CPU启动后总计数
st = 'E:\\1裂缝检测\\testdata\\vedio_Finish\\10'
# for x in file_paths:
if 1:
	x = file_paths[7]
	videoCapture = cv2.VideoCapture(x)  # 捕捉视频，未开始读取；
	fps = videoCapture.get(cv2.CAP_PROP_FPS)  # 获取视频帧速率
	size = (int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)),
	        int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))  # 获取视频尺寸 
	videoWrite = cv2.VideoWriter(st + ".avi",
	                             cv2.VideoWriter_fourcc("I", "4", "2", "0"), fps, size)
	st += '1'
	success, frame = videoCapture.read()  # 读帧
	while success:  # Loop until there are no more frames.
		# cv2.imshow("zd1", frame)
		frame = classify(frame,imgNum)
		# cv2.imshow("zd1", frame)
		# cv2.waitKey(int(1000/fps))  # 1000毫秒/帧速率
		videoWrite.write(frame)  # 写视频帧
		success, frame = videoCapture.read()  # 获取下一帧
	print(x) 
	t2 = cv2.getTickCount()
	print((t2-t1)/cv2.getTickFrequency())  # cv2.getTickFrequency()返回CPU频率（每秒计数）
	# break
