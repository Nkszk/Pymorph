import cv2
import numpy as np
import morph
import time

#Image
img = cv2.imread("IMG/Lena512_noi_s25.png",0)

#Parameters
params = np.load("./Param/uint8_5_24_15_M.npy", allow_pickle=True) #for 5x5 SEs
#params = np.load("./Param/uint8_7_48_15_M.npy", allow_pickle=True) #for 7x7 SEs


se    = params[0]
kappa = params[1]

NS = se.shape[0] # Size of structuring elements
K  = se.shape[2] # Number of subnet pairs
L  = se.shape[3]

cv2.imshow('in',img)

vi     = np.zeros([img.shape[0], img.shape[1]], np.uint8)
temp1  = np.zeros([img.shape[0], img.shape[1]], np.uint8)
temp2  = np.zeros([img.shape[0], img.shape[1]], np.uint8)
avei   = np.zeros([img.shape[0], img.shape[1]], np.uint8)
vi     = np.array(img, np.uint8)

temp_se = np.zeros([NS, NS, K], np.uint8)

t1 = time.time()

for i in range(L):
	temp_se = np.array(se[:, :, :, i], np.uint8)

	morph.modmoe_5x5(vi, avei, temp_se)
#	morph.modmoe_7x7(vi, avei, temp_se)
	
	vi_f = (1.0 - 2.0 * kappa[i]) * vi + 2.0 * kappa[i] * avei + 0.5
	vi_f[vi_f>255.0] = 255.0
	vi_f[vi_f<0] = 0.0

	vi = np.array(vi_f, np.uint8)
	
t2 = time.time()
elapsed_time = t2 -t1
print(f"経過時間：{elapsed_time}")

cv2.imshow('out',vi )
cv2.waitKey(0)
cv2.destroyWindow('in')
cv2.destroyWindow('out')
