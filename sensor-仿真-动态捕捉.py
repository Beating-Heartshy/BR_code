import cv2
import numpy as np
import math
from PIL import Image
sensor_list = []
Last_list = []
max_series_days = 1
New_list = []
width = 640  # 设置传感器宽度
height = 360  # 设置传感器高度
block_size = 2 #分辨率的块大小
width = int(width)
height = int(height)
block_size = int(block_size)

Rezult = np.zeros((height//block_size, width//block_size))
t = np.zeros((height//block_size, width//block_size))
L2 = np.zeros((height//block_size, width//block_size))
def zhuan (im):
    im_array = np.array(im)

    return (im_array)

def sensor (im_array):

    V = 5 #设置阈值


    # print('im_array',im_array.shape)
    im_array2 = cv2.cvtColor(im_array.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    #归一化
    # im_array2 = im_array2 / 255
    # 获取数组的形状
    shape = im_array2 .shape
    # print(shape)

    # 定义块的大小和数量

    blocks_per_row = shape[1] // block_size
    blocks_per_col = shape[0] // block_size
    # 创建一个新的数组来保存每个块的平均值
    mean_blocks = np.zeros((blocks_per_col, blocks_per_row))
    for i in range(blocks_per_col):
        for j in range(blocks_per_row):
            block = im_array2[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size]
            mean = np.mean(block)
            mean_blocks[i, j] = mean


    # print('mean_blocks',mean_blocks.shape)
    sensor_list.append(mean_blocks)
    if len(sensor_list) > max_series_days:
        Last_list.append(sensor_list[0])
        New_list.append(sensor_list[1])
        sensor_list.pop(0)  # 移除最左侧的元素
        Last = np.array(Last_list)
        New = np.array(New_list)
        # 删除第一个维度
        Last = np.squeeze(Last, axis=0)
        New = np.squeeze(New, axis=0)
        # print(Last.shape)
        # print(New.shape)
        for i in range(blocks_per_col):
            for j in range(blocks_per_row):
                if New[i,j]-Last[i,j]>V or New[i,j]-Last[i,j]<-V:
                # if New[i,j]-Last[i,j]<-V :
                    Rezult[i,j] = New[i,j]
                    t[i,j] = 0
                    L2[i,j] = New[i,j]
                else:
                    t[i,j] += 1
                    Rezult[i,j] = L2[i,j]*math.exp(-0.05*t[i,j])
        Last_list.clear()
        New_list.clear()
        Last.fill(0)
        New.fill(0)
    else:
        for i in range(blocks_per_col):
            for j in range(blocks_per_row):
                Rezult[i,j] = mean_blocks[i,j]
                L2[i,j] = mean_blocks[i,j]








# 打开视频文件
cap = cv2.VideoCapture('55.mp4')  # 替换成你的视频文件路径

fps = cap.get(cv2.CAP_PROP_FPS)
# 检查视频是否成功打开
if not cap.isOpened():
    print("Error opening video file")
    exit()

# # 创建视频写入对象
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # 视频编解码器
# out = cv2.VideoWriter('output_video.avi', fourcc, 30, (640, 480))  # 输出视频文件名和调整后的宽高
out = cv2.VideoWriter('LJJ0.05.avi', fourcc, 30, (width, height), isColor=False)

# # 创建窗口
# cv2.namedWindow('Processed Video', cv2.WINDOW_NORMAL)
# 循环读取每一帧
while True:
    # 从视频中读取一帧
    ret, frame = cap.read()

    # 检查是否成功读取帧
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # 在这里对每一帧进行处理，比如显示、保存等
    # 调整图像大小为640x480
    # new_size = (640, 480)
    # resized_image = frame.resize(new_size)
    # print(resized_image)
    # 调整图像大小为高480、宽640
    resized_frame = cv2.resize(frame, (width, height))
    # 检查是否存在 None 值
    if resized_frame is None:
        print("Error: Image array is None")
    else:
        sensor(zhuan(resized_frame))
        gray_img = np.array(Rezult, dtype=np.uint8)
        # 根据像素值生成颜色映射表（伪彩色图像）
        # colormap = cv2.applyColorMap(gray_img, cv2.COLORMAP_JET)


        # cv2.destroyAllWindows()  # 将数组转换为灰度图像
        # gray_img = np.array(Rezult, dtype=np.uint8)
        # 将图像放大20倍
        # resized_img = cv2.resize(colormap, (colormap.shape[1] * 20, colormap.shape[0] * 20), interpolation=cv2.INTER_CUBIC)
        resized_img = cv2.resize(gray_img, (gray_img.shape[1] * block_size, gray_img.shape[0] * block_size),
                                 interpolation=cv2.INTER_CUBIC)
        # cv2.imshow('Processed Video', resized_img)  # 显示图像
        out.write(resized_img)

        # cv2.imshow('Processed Video', resized_img)  # 显示图像
    # key = cv2.waitKey(int(2000 / fps))  # 通过帧率计算等待时间，保持实时播放




    # 按下 'q' 键退出循环
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# 释放视频对象和关闭窗口
cap.release()
out.release()
cv2.destroyAllWindows()
