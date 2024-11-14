import  cv2
import numpy as np
import math
import time
import copy


def resize_with_gus(raw_img,dst_shape):
    cur_img = raw_img.copy()
    dst_h, dst_w = dst_shape[0], dst_shape[1]
    while True:
        cur_h, cur_w,_ = cur_img.shape
        if cur_h==dst_h and cur_w ==dst_w:
            return cur_img
        cur_img = reszie_with_gus_2x(cur_img,dst_shape)

def reszie_with_gus_2x(raw_img,dst_shape):
    ori_h, ori_w,_ = raw_img.shape
    dst_h, dst_w = dst_shape[0], dst_shape[1]
    aim_h ,aim_w = max(int(ori_h/2),dst_h),max(int(ori_w/2),dst_w)
    ksize_width = int(ori_h/aim_h)|1
    ksize_height = int(ori_w/aim_w)|1
    sigmaX = ori_w/aim_w*0.6
    sigmaY = ori_h/aim_h*0.6
    # print(f"ksize_width:{ksize_width},ksize_height:{ksize_height},sigmaX:{sigmaX},sigmaY:{sigmaY}\n")
    blurred_img = cv2.GaussianBlur(raw_img, (ksize_width, ksize_height), sigmaX, sigmaY)
    return cv2.resize(blurred_img,(aim_w,aim_h))
    


def low_pass_filter(image, kernel_size=(5, 5), sigma=1.0):
    start_time = time.time()
    # 创建高斯核
    kernel = cv2.getGaussianKernel(kernel_size[0], sigma)
    kernel = kernel * np.transpose(kernel)
    kernel = kernel / np.sum(kernel)  # Normalize the kernel
    
    # 使用卷积核对每个通道进行卷积操作
    channels = cv2.split(image)
    filtered_channels = [cv2.filter2D(c, -1, kernel) for c in channels]
    filtered_image = cv2.merge(filtered_channels)
    end_time = time.time()-start_time
    print(f"once lowpass cost {end_time} sceonds")
    return filtered_image

def get_sigma(ori_shape, dst_shape):
    ori_h, ori_w = ori_shape[0], ori_shape[1]
    dst_h, dst_w = dst_shape[0], dst_shape[1]
    scale_w = ori_w/dst_w
    scale_h = ori_h/dst_h
    # 使用最小值确保sigma不会过大
    sigma = min((scale_w + scale_h), 10)  # 限制sigma的最大值
    return int(sigma/4)

def get_kernel(ori_shape, dst_shape):
    ori_h, ori_w = ori_shape[0], ori_shape[1]
    dst_h, dst_w = dst_shape[0], dst_shape[1]
    w_kernel = int(np.ceil(dst_w / ori_w))
    h_kernel = int(np.ceil(dst_h / ori_h))
    # 确保核大小至少为3，且为奇数
    return (max(w_kernel | 1, 3), max(h_kernel | 1, 3))

def lowpass(image, dst_shape):
    ori_shape = image.shape[:-1]
    kernel_size = get_kernel(ori_shape, dst_shape)
    sigma = get_sigma(ori_shape, dst_shape) 
    print(f"kernel size {kernel_size};sigma {sigma}")
    ret =  low_pass_filter(image, kernel_size, sigma)  
    return ret