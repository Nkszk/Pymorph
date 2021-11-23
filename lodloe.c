#include <stdio.h>
#include <stdlib.h>
#include <arm_neon.h>

#define NS5 5 //Size of SE 5x5
#define NC5 2 //Center of SE 5x5

#define NS7 7 //Size of SE 7x7
#define NC7 3 //Center of SE 7x7

void lod_simd_5x5(uint8_t input_data[], uint8_t output_data[], uint8_t se[], float32_t alpha[], int H, int W, int K)
{
	int sx, sy;
	int i, j, x, y, xs, ys, z;
	uint8x8_t pixel4[NS5*NS5]; //Set of pixels supported by SE
	uint8x8_t temp_max;
	uint8x8_t pixel;
	uint8x8_t se_element;
	uint8x8_t out_d;
	uint8x8_t zero8;
	uint8x8_t max8;

	uint8_t czero[8] = {  0,   0,   0,   0,   0,   0,   0,   0};
	uint8_t cmax[8]  = {255, 255, 255, 255, 255, 255, 255, 255};

    float32_t acc[8];
	uint8_t   i_temp_max[8];
	uint8_t   round_acc[8];

  //Allocation Memory for the block
  uint8_t *block;
  block = (uint8_t *)alloca( sizeof(uint8_t) * NS5 * (NC5 + W + NC5) );

	zero8 = vld1_u8(czero);
	max8  = vld1_u8(cmax);

	for(y=0; y<NC5; y++) //Initialization of the block, 0 to NC
	{
		for(x=0; x < 2*NC5+W; x++)
		{
			block[(2*NC5+W) * y + x] = 0;
		}
	}
	for(y=NC5; y<NS5; y++) //Initialization of the block, NC to NS 
	{
		for(x=    0; x <     NC5; x++) block[(2*NC5+W) * y + x] = 0;
		for(x=  NC5; x <   NC5+W; x++) block[(2*NC5+W) * y + x] = input_data[(y-NC5)*W + (x - NC5)];
		for(x=NC5+W; x < 2*NC5+W; x++) block[(2*NC5+W) * y + x] = 0;
	}

	for(y=0; y<H; y++)
	{
		for(x=0; x<W; x=x+8)
		{
			//The block to pixel4
			for(xs=-NC5; xs<=NC5; xs++)
			{
				for(ys=-NC5; ys<=NC5; ys++)
				{
					pixel4[NS5 *(ys+NC5) + (xs+NC5)] = vld1_u8(&(block[(2*NC5+W) * (NC5+ys) + (x+xs+NC5)]));
				}
			}

			acc[0] = 0.0; acc[1] = 0.0; acc[2] = 0.0; acc[3] = 0.0; acc[4] = 0.0; acc[5] = 0.0; acc[6] = 0.0; acc[7] = 0.0; //Clear

			for(j=0; j<K; j++)
			{
				temp_max = zero8; //Clear max as zero
				for(sx=0; sx<NS5*NS5; sx=sx+1)
				{
					se_element = vdup_n_u8(se[sx * K + j]); //Copy a bias to each lane
					pixel      = vqsub_u8(pixel4[sx], se_element); //Saturated addition
					temp_max   = vmax_u8(temp_max, pixel); //Max
				}
				vst1_u8(i_temp_max, temp_max); 
				//Linear combination
				acc[0] = acc[0] + (float32_t)i_temp_max[0] * alpha[j];
				acc[1] = acc[1] + (float32_t)i_temp_max[1] * alpha[j];
				acc[2] = acc[2] + (float32_t)i_temp_max[2] * alpha[j];
				acc[3] = acc[3] + (float32_t)i_temp_max[3] * alpha[j];
				acc[4] = acc[4] + (float32_t)i_temp_max[4] * alpha[j];
				acc[5] = acc[5] + (float32_t)i_temp_max[5] * alpha[j];
				acc[6] = acc[6] + (float32_t)i_temp_max[6] * alpha[j];
				acc[7] = acc[7] + (float32_t)i_temp_max[7] * alpha[j];
			}
			//Rounding, quantization
			for(z=0; z<8; z++){
				if(acc[z] > 254.5)acc[z]=255;
				else if(acc[z] < 0.0)acc[z]=0;
				else acc[z]=acc[z]+0.5;
			}
			round_acc[0] = (uint8_t)acc[0];
			round_acc[1] = (uint8_t)acc[1];
			round_acc[2] = (uint8_t)acc[2];
			round_acc[3] = (uint8_t)acc[3];
			round_acc[4] = (uint8_t)acc[4];
			round_acc[5] = (uint8_t)acc[5];
			round_acc[6] = (uint8_t)acc[6];
			round_acc[7] = (uint8_t)acc[7];
			out_d = vld1_u8(round_acc);
			
			vst1_u8(&(output_data[y*W+x]), out_d); //Store
		}

		//Update the block
		for(ys=0; ys<NS5-1; ys++)
		{
			for(xs=0; xs<2*NC5+W; xs++)
			{
				block[(2*NC5+W)*ys + xs] = block[(2*NC5+W)*(ys+1) + xs];
			}
		}
		if((y+NC5+1) < H)
		{
			for(xs=    0; xs <     NC5; xs++) block[(2*NC5+W)*(NS5 - 1) + xs] = 0;
			for(xs=  NC5; xs <   NC5+W; xs++) block[(2*NC5+W)*(NS5 - 1) + xs] = input_data[W * (y + NC5 + 1) + xs - NC5];
			for(xs=NC5+W; xs < 2*NC5+W; xs++) block[(2*NC5+W)*(NS5 - 1) + xs] = 0;
		}
		else
		{
			for(xs=0; xs< 2*NC5+W; xs++) block[(2*NC5+W)*(NS5 - 1) + xs] = 0;
		}
	}

}

void loe_simd_5x5(uint8_t input_data[], uint8_t output_data[], uint8_t se[], float32_t alpha[], int H, int W, int K)
{
	int sx, sy;
	int i, j, x, y, xs, ys, z;
	uint8x8_t pixel4[NS5*NS5]; //Set of pixels supported by SE
	uint8x8_t temp_min;
	uint8x8_t pixel;
	uint8x8_t se_element;
	uint8x8_t out_d;
	uint8x8_t zero8;
	uint8x8_t max8;
	uint8_t czero[8] = {  0,   0,   0,   0,   0,   0,   0,   0};
	uint8_t  cmax[8] = {255, 255, 255, 255, 255, 255, 255, 255};
	float32_t acc[8];
	uint8_t i_temp_min[8];
	uint8_t round_acc[8];

	zero8 = vld1_u8(czero);
	max8  = vld1_u8(cmax);

	//Allocation Memory for the block
	uint8_t *block;
	block = (uint8_t *)alloca( sizeof(uint8_t) * NS5 * (NC5 + W + NC5) );

	for(y=0; y<NC5; y++) //Initialization of the block, 0 to NC
	{
		for(x=0; x < 2*NC5+W; x++)
		{
			block[(2*NC5+W) * y + x] = 255;
		}
	}
	for(y=NC5; y<NS5; y++) //Initialization of the block, NC to NS 
	{
		for(x=   0; x <      NC5; x++) block[(2*NC5+W) * y + x] = 255;
		for(x=  NC5; x <   NC5+W; x++) block[(2*NC5+W) * y + x] = input_data[(y-NC5) * W + (x - NC5)];
		for(x=NC5+W; x < 2*NC5+W; x++) block[(2*NC5+W) * y + x] = 255;
	}

	for(y=0; y<H; y++)
	{
		for(x=0; x<W; x=x+8)
		{
			//The block to pixel4
			for(xs=-NC5; xs<=NC5; xs++)
			{
				for(ys=-NC5; ys<=NC5; ys++)
				{
					pixel4[NS5 *(ys+NC5) + (xs+NC5)] = vld1_u8(&(block[(2*NC5+W) * (NC5+ys) + (x+xs+NC5)]));
				}
			}

			acc[0] = 0.0; acc[1] = 0.0; acc[2] = 0.0; acc[3] = 0.0; acc[4] = 0.0; acc[5] = 0.0; acc[6] = 0.0; acc[7] = 0.0; //0でクリア
			for(j=0; j<K; j++)
			{
				temp_min = max8; //Clear min as 255
				for(sx=0; sx<NS5*NS5; sx=sx+1)
				{
					se_element = vdup_n_u8(se[(NS5*NS5-1-sx) * K + j]); //Copy a bias to each lane
					pixel = vqadd_u8(pixel4[sx], se_element); //Saturated addition
					temp_min = vmin_u8(temp_min, pixel); //Min
				}
				vst1_u8(i_temp_min, temp_min); 
				//Linear combination
				acc[0] = acc[0] + (float32_t)i_temp_min[0] * alpha[j];
				acc[1] = acc[1] + (float32_t)i_temp_min[1] * alpha[j];
				acc[2] = acc[2] + (float32_t)i_temp_min[2] * alpha[j];
				acc[3] = acc[3] + (float32_t)i_temp_min[3] * alpha[j];
				acc[4] = acc[4] + (float32_t)i_temp_min[4] * alpha[j];
				acc[5] = acc[5] + (float32_t)i_temp_min[5] * alpha[j];
				acc[6] = acc[6] + (float32_t)i_temp_min[6] * alpha[j];
				acc[7] = acc[7] + (float32_t)i_temp_min[7] * alpha[j];
			}
			//Rounding, quantization
			for(z=0; z<8; z++){
				if(acc[z] > 254.5)acc[z]=255;
				else if(acc[z] < 0.0)acc[z]=0;
				else acc[z]=acc[z]+0.5;
			}
			round_acc[0] = (uint8_t)acc[0];
			round_acc[1] = (uint8_t)acc[1];
			round_acc[2] = (uint8_t)acc[2];
			round_acc[3] = (uint8_t)acc[3];
			round_acc[4] = (uint8_t)acc[4];
			round_acc[5] = (uint8_t)acc[5];
			round_acc[6] = (uint8_t)acc[6];
			round_acc[7] = (uint8_t)acc[7];

			vst1_u8(&(output_data[y*W+x]), out_d); //Store
		}
		//Update the block
		for(ys=0; ys<NS5-1; ys++)
		{
			for(xs=0; xs<2*NC5+W; xs++)
			{
				block[(2*NC5+W)*ys + xs] = block[(2*NC5+W)*(ys+1) + xs];
			}
		}
		if((y+NC5+1) < W)
		{
			for(xs=   0; xs <      NC5; xs++) block[(2*NC5+W)*(NS5 - 1) + xs] = 255;
			for(xs=  NC5; xs <   NC5+W; xs++) block[(2*NC5+W)*(NS5 - 1) + xs] = input_data[W * (y + NC5 + 1) + xs - NC5];
			for(xs=NC5+W; xs < 2*NC5+W; xs++) block[(2*NC5+W)*(NS5 - 1) + xs] = 255;
		}
		else
		{
			for(xs=   0; xs < 2*NC5+W; xs++) block[(2*NC5+W)*(NS5 - 1) + xs] = 255;
		}
	}
}
