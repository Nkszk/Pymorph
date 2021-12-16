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
sudo python3 setup.py install
```
4. Run Gaussian denoising by the deep Gaussian denoser [^1][^2]  (Noise standard deviation sigma = 25, 512x512 image will be processed by about  1.8 [s]. )
```
python3 morph_gauss.py
```
5. Run image completion example [^3] (Missing probability = 0.7)
```
python3 morph_int.py
```

## Extended moprhological filters
In module "pymorph", structuring elemebnts are defined by three dimentinal array as
structuring_elements[index, x, y]
where the set of x, y denotes the corrdinates. Images and structring elements are limited in uint8. 

**Maximum of erosions (MoE)**: the pixel-width maximum of the set of eroded images.

 pymorph.moe5x5 (source, dist, structuring_elements) 

 (structuring_elements is defined in the size 5x5.)

 pymorph.moe7x7 (source, dist, structuring_elements)

 (structuring_elements is defined in the size 7x7.)

###　Maximum of dilations (MoD): the pixel-width maximum of the set of eroded images.

 pymorph.mod5x5 (source, dist, structuring_elements) 

 (structuring_elements is defined in the size 5x5.)

 pymorph.mod7x7 (source, dist, structuring_elements)

 (structuring_elements is defined in the size 7x7.)
 
###　Linear combination of erosions (LoE) : the linear combination of the set of eroded images.

 pymorph.loe5x5 (source, dist, structuring_elements, weights) 

 (structuring_elements is defined in the size 5x5. weights are float32)

###　Linear combination of dilations (LoD) : the linear combination of the set of dilated images.

 pymorph.lod5x5 (source, dist, structuring_elements, weights) 

 (structuring_elements is defined in the size 5x5. weights are float32)

## Training of parameters
All paremeters are trained by TensorFlow. The details of the training are explained in [^2] and [^3].

[^1]: H. Fujisaki, M. Nakashizuka, "Deep Morphological Filter Networks For Gaussian Denoising, " Proc. 2020 IEEE International Conference on Image Processing, pp. 918-922, Abu Dhabi, Oct. 2020.
[^2]: H. Fujisaki, M, Nakashizuka, "Deep Gaussian denoising network based on morphological operators with low-precision arithmetic, " IEICE Trans. on Fundamentals, Vol.E105-A,No.4,pp.-,Apr. 2022.
[^3]: G. Okada, S. Nozawa and M. Nakashizuka, "Morphological Operators with Multiple Structuring Elements and Its Training for Image Denoising and Completion," 2018 International Symposium on Intelligent Signal Processing and Communication Systems (ISPACS), pp. 406-410, Oct. 2018.
