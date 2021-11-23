#!/usr/bin/env python
# -*- coding: utf-8 -*-

#Image completion by averaging of extended erosion and dilation

import cv2
import numpy as np
import morph
import time

#Image read
input_image = cv2.imread("IMG/lena.png", cv2.IMREAD_GRAYSCALE) #Original image
mask_image  = cv2.imread("IMG/mask_lena_750.png", cv2.IMREAD_GRAYSCALE) #Image mask

input_image = np.array(input_image)
mask_image  = np.array(mask_image)

se_e = np.load("Param/Int_p_080/e_7x7_32_080_uint8_param.npy")
se_d = np.load("Param/Int_p_080/d_7x7_32_080_uint8_param.npy")

ximg  = input_image
xmask = mask_image

#Generate mask
mask_tf = (mask_image == 255)
mask = np.array( mask_tf.astype('uint8') )

#Inputs
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

t1 = time.time()
for k in range(1):
	morph.mod_7x7(in_img_d, out_img_d, se_d)
	morph.moe_7x7(in_img_e, out_img_e, se_e)
	morph.ave(out_img_d, out_img_e, out)
	out = (1 - mask) * out + mask * input_image

#out = out.astype("uint8")
t2 = time.time()
elapsed_time = t2 -t1
print(f"Elapsed time: {elapsed_time/1}")

cv2.imshow('out', out)
cv2.waitKey(0)
cv2.destroyAllWindows()
