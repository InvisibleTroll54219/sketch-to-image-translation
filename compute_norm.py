from skimage.metrics import structural_similarity
import numpy as np
import natsort
import os
import cv2

def load_filename(path):
    dirFiles = os.listdir(path)
    for i, file in enumerate(dirFiles):
        dirFiles[i] = path + file
    return natsort.natsorted(dirFiles ,reverse=False)

def load_images(list_path):
    img_list = list()
    for filename in list_path:
        pixels = cv2.imread(filename)
        img_list.append(pixels)
    return np.asarray(img_list)

def compute_l2(imgs1, imgs2):
    l2_scores = []
    for i in range(len(imgs1)):
        score = (np.square(imgs1[i] - imgs2[i])).mean()
        l2_scores.append(score)
    return np.mean(l2_scores)

def compute_ssim(imgs1, imgs2):
    ssim_scores = []
    for i in range(len(imgs1)):
        grayA = cv2.cvtColor(imgs1[i], cv2.COLOR_BGR2GRAY)
        grayB = cv2.cvtColor(imgs2[i], cv2.COLOR_BGR2GRAY)
        (score, diff) = structural_similarity(grayA, grayB, full=True)
        ssim_scores.append(score)
    return np.mean(score)

imgs1 = load_images(load_filename("Dataset/CUHK/Testing photo/"))
imgs2 = load_images(load_filename("Generated Images/Generated_Pixel[1]_Context[0]/"))

l2 = compute_l2(imgs1, imgs2)
ssim = compute_ssim(imgs1, imgs2)

print("Pixel loss weight : 1 - Contextual loss weight : 0 => L2-norm: " + str(l2) + " :: SSIM: " + str(ssim))

imgs1 = load_images(load_filename("Dataset/CUHK/Testing photo/"))
imgs2 = load_images(load_filename("Generated Images/Generated_Pixel[08]_Context[02]/"))

l2 = compute_l2(imgs1, imgs2)
ssim = compute_ssim(imgs1, imgs2)

print("Pixel loss weight : 0.8 - Contextual loss weight : 0.2 => L2-norm: " + str(l2) + " :: SSIM: " + str(ssim))

imgs1 = load_images(load_filename("Dataset/CUHK/Testing photo/"))
imgs2 = load_images(load_filename("Generated Images/Generated_Pixel[05]_Context[05]/"))

l2 = compute_l2(imgs1, imgs2)
ssim = compute_ssim(imgs1, imgs2)

print("Pixel loss weight : 0.5 - Contextual loss weight : 0.5 => L2-norm: " + str(l2) + " :: SSIM: " + str(ssim))

imgs1 = load_images(load_filename("Dataset/CUHK/Testing photo/"))
imgs2 = load_images(load_filename("Generated Images/Generated_Pixel[02]_Context[08]/"))

l2 = compute_l2(imgs1, imgs2)
ssim = compute_ssim(imgs1, imgs2)

print("Pixel loss weight : 0.2 - Contextual loss weight : 0.8 => L2-norm: " + str(l2) + " :: SSIM: " + str(ssim))