#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import morph
import time

se = np.load("MOD_SE_learnt_750_49.npy")
se = -1.0 * se
m = se < 0
m = m.astype(float)
se = se - m * se
uint8_se = se.astype("uint8")

ximg = cv2.imread("man_pepper_750.pgm",0)
img = np.array(255 - ximg[0:512, 0:512], np.uint8)


out_img = np.zeros((img.shape[0], img.shape[1]), np.uint8)
in_img  = np.zeros((img.shape[0], img.shape[1]), np.uint8)

in_img = img;

cv2.imshow('in',in_img)


t1 = time.time()
morph.mod_7x7(in_img, out_img, uint8_se)
t2 = time.time()
elapsed_time = t2 -t1
print(f"経過時間：{elapsed_time}")

cv2.imshow('out',255 - out_img)
cv2.waitKey(0)
cv2.destroyAllWindows()