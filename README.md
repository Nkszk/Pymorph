# Pymorph
Extended Morphological filter library using ARM SIMD instructions for Python on Raspberry Pi.
Deep network for Gaussain denoiser and image completion examples using the library are also provided.
This library and image processing applications were implemented to demostrate the fast computation in the papers [^1][^2][^3].

This library is tested on Raspbian 10, Raspberry pi 4. This library will work on other ARM systems with NEON SIMD instruction.

## Install
1. Clone code and example images:
```
git clone https://github.com/Nkszk/Pymorph
```
2. Install python3-dev
```
sudo apt-get install python3-dev
```
3. Compile and install
```
cd Pymorph
gcc -c lodloe.c -mfpu=neon -march=armv7-a -O3
gcc  -c modmoe.c -mfpu=neon -march=armv7-a -O3
sudo apt-get install python3-dev
```
4. Run Gaussian denoising by the deep Gaussian denoser [^1][^2]  (Noise standard deviation sigma = 25, 512x512 image will be processed by about  1.8 [s]. )
```
python3 morph_gauss.py
```
5. Run image completion example [^3] (Missing probability = 0.7)
```
python3 morph_int.py
```

## About extended moprhological filters

## About training of parameters

[^1]: H. Fujisaki, M. Nakashizuka, "Deep Morphological Filter Networks For Gaussian Denoising, " Proc. 2020 IEEE International Conference on Image Processing, pp. 918-922, Abu Dhabi, Oct. 2020.
[^2]: H. Fujisaki, M, Nakashizuka, "Deep Gaussian denoising network based on morphological operators with low-precision arithmetic, " IEICE Trans. on Fundamentals, Vol.E105-A,No.4,pp.-,Apr. 2022.
[^3]: G. Okada, S. Nozawa and M. Nakashizuka, "Morphological Operators with Multiple Structuring Elements and Its Training for Image Denoising and Completion," 2018 International Symposium on Intelligent Signal Processing and Communication Systems (ISPACS), pp. 406-410, Oct. 2018.
