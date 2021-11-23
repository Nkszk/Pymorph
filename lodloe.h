#ifndef  _LODLOE_H_
#define  _LODLOE_H_
//Linear combination of dilations 
void lod_simd_5x5(uint8_t[], uint8_t[], uint8_t[], float32_t[], int, int, int);
//Linear combination of erosions
void loe_simd_5x5(uint8_t[], uint8_t[], uint8_t[], float32_t[], int, int, int);
#endif //_LODLOE_H_
