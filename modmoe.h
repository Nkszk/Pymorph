#ifndef  _MODMOE_H_
#define  _MODMOE_H_
void dilation_simd_5x5(uint8_t[], uint8_t[], uint8_t[], int, int, int);
void dilation_simd_7x7(uint8_t[], uint8_t[], uint8_t[], int, int, int);
void erosion_simd_5x5(uint8_t[], uint8_t[], uint8_t[], int, int, int);
void erosion_simd_7x7(uint8_t[], uint8_t[], uint8_t[], int, int, int);
void ave(uint8_t[], uint8_t[], uint8_t[], int, int);
#endif //_MODMOE_H_
