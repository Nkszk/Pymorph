#ifndef  _MODMOE_H_
#define  _MODMOE_H_
//Minimum of Dilations 5x5
void dilation_simd_5x5(uint8_t[], uint8_t[], uint8_t[], int, int, int);
//Minimum of Dilations 7x7
void dilation_simd_7x7(uint8_t[], uint8_t[], uint8_t[], int, int, int);
//Maximum of Erosions 5x5
void erosion_simd_5x5(uint8_t[], uint8_t[], uint8_t[], int, int, int);
//Maximum of Erosions 5x5
void erosion_simd_7x7(uint8_t[], uint8_t[], uint8_t[], int, int, int);
//Rounding averaging
void ave(uint8_t[], uint8_t[], uint8_t[], int, int);
#endif //_MODMOE_H_
