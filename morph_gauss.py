#!/usr/bin/env python
# -*- coding: utf-8 -*-
import cv2
import numpy as np
import morph
import time

#Image
img = cv2.imread("IMG/Lena512_noi_s25.png",0)

#Parameters
se_d_r = np.load("./Param/Gaussian_sigma_25/d.npy")
se_e_r = np.load("./Param/Gaussian_sigma_25/e.npy")
alpha  = np.load("./Param/Gaussian_sigma_25/a.npy")

wod_r  = np.load("./Param/Gaussian_sigma_25/w_od.npy")
woe_r  = np.load("./Param/Gaussian_sigma_25/w_oe.npy")
wcd_r  = np.load("./Param/Gaussian_sigma_25/w_cd.npy")
wce_r  = np.load("./Param/Gaussian_sigma_25/w_ce.npy")

vi     = np.array(img, np.uint8)
temp_1 = np.zeros([img.shape[0], img.shape[1]], np.uint8)
temp_2 = np.zeros([img.shape[0], img.shape[1]], np.uint8)
vo_o   = np.zeros([img.shape[0], img.shape[1]], np.uint8)
vo_c   = np.zeros([img.shape[0], img.shape[1]], np.uint8)
vv     = np.zeros([img.shape[0], img.shape[1]], np.uint8)
vi_f   = np.zeros([img.shape[0], img.shape[1]], np.float32)

#Network configuration
NS = se_d_r.shape[0] # Size of structuring elements
K  = se_d_r.shape[2] # Number of subnet pairs
L  = se_d_r.shape[3] # Number of Stages

temp_se = np.zeros([NS, NS, K], np.uint8)
temp_w  = np.zeros(K, np.float32)

cv2.imshow('Noisy',img)

t1 = time.time()
for ll in range(L):
    temp_se = np.array(se_e_r[:, :, :, ll], np.uint8)
    temp_w  = np.array(woe_r[:, ll], np.float32)
    morph.loe_5x5(vi, temp_1, temp_se, temp_w)

    temp_se = np.array(se_d_r[:, :, :, ll], np.uint8)
    temp_w  = np.array(wcd_r[:, ll], np.float32)
    morph.lod_5x5(vi, temp_2, temp_se, temp_w)

    temp_se = np.array(se_d_r[:, :, :, ll], np.uint8)
    temp_w  = np.array(wod_r[:, ll], np.float32)
    morph.lod_5x5(temp_1, vo_o, temp_se, temp_w)

    temp_se = np.array(se_e_r[:, :, :, ll], np.uint8)
    temp_w  = np.array(wce_r[:, ll], np.float32)
    morph.loe_5x5(temp_2, vo_c, temp_se, temp_w)

    morph.ave(vo_o, vo_c, vv)

    vi_f = (1.0 - alpha[ll]) * vi + alpha[ll] * vv + 0.5
    vi_f[vi_f>255.0] = 255.0
    vi_f[vi_f<0] = 0.0
    vi = np.array(vi_f, np.uint8)
t2 = time.time()
elapsed_time = t2 -t1
print(f"Elapsed time：{elapsed_time}")

cv2.imshow('Denoise',vi )
cv2.waitKey(0)
cv2.destroyWindow('Noisy')
cv2.destroyWindow('Denoise')
