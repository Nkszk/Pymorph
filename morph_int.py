#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import morph
import time

#画像の読み込み
input_image = cv2.imread("../../Images/lena.png", cv2.IMREAD_GRAYSCALE) #画像の読み込み, このプログラムと同じフォルダに画像をいれること．
mask_image  = cv2.imread("../../Images/mask_lena_750.png", cv2.IMREAD_GRAYSCALE) #画像の読み込み, このプログラムと同じフォルダに画像をいれること．

input_image = np.array(input_image)
mask_image  = np.array(mask_image)

se_e = np.load("./Param/Int_p_075/ML_sig_se_e_learnt_750_49.npy")
se_e = 255.0 /(1 + np.exp(-se_e))

se_d = np.load("./Param/Int_p_075/ML_sig_se_d_learnt_750_49.npy")
se_d = 255.0 /(1 + np.exp(-se_d))

m = se_e > 255
m = m.astype(float)
se_e = (1.0 - m) * se_e + m * 255
m = se_e < 0
m = m.astype(float)
se_e = se_e - m * se_e
uint8_se_e = se_e.astype("uint8")

m = se_d > 255
m = m.astype(float)
se_d = (1.0 - m) * se_d + m * 255
m = se_d < 0
m = m.astype(float)
se_d = se_d - m * se_d
uint8_se_d = se_d.astype("uint8")

ximg  = input_image
xmask = mask_image

#マスクの生成
mask_tf = (mask_image == 255)
mask = np.array( mask_tf.astype('uint8') )

#欠損
input_image_d = mask * input_image + (1 - mask) * 0
input_image_e = mask * input_image + (1 - mask) * 255

out_img_d = np.zeros((mask.shape[0], mask.shape[1]), np.uint8)
out_img_e = np.zeros((mask.shape[0], mask.shape[1]), np.uint8)
out       = np.zeros((mask.shape[0], mask.shape[1]), np.uint8)

in_img_e  = np.zeros((mask.shape[0], mask.shape[1]), np.uint8)
in_img_d  = np.zeros((mask.shape[0], mask.shape[1]), np.uint8)

in_img_e = input_image_e
in_img_d = input_image_d

cv2.imshow('in',in_img_d)
cv2.waitKey(0)

t1 = time.time()
for k in range(1):
	morph.mod_7x7(in_img_d, out_img_d, uint8_se_e)
	morph.moe_7x7(in_img_e, out_img_e, uint8_se_d)
	morph.ave(out_img_d, out_img_e, out)
	out = (1 - mask) * out + mask * input_image

#out = out.astype("uint8")
t2 = time.time()
elapsed_time = t2 -t1
print(f"経過時間：{elapsed_time/1}")

cv2.imshow('out', out)
cv2.waitKey(0)
cv2.destroyAllWindows()
