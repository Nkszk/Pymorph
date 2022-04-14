#include <stdio.h>
#include <stdlib.h>
#include <arm_neon.h>

#define NS5 5 //5x5の場合の構造要素のサイズ
#define NC5 2 //5x5の場合の構造要素の中心

#define NS7 7 //7x7の場合の構造要素のサイズ
#define NC7 3 //7x7の場合の構造要素の中心

void dilation_simd_5x5(uint8_t input_data[], uint8_t out_data[], uint8_t se[], int H, int W, int K)
{
	// H: 画像の垂直方向の画素数, W: 画像の水平方向への画素数, K: 構造要素の数
	int sx, sy;
	int i, j, x, y, xs, ys;
	uint8x16_t temp_max; //SIMD用のレジスタ
	uint8x16_t pixel8[NS5*NS5]; //まずは，メモリからローカルな変数へコピーするための配列
	uint8x16_t pixel;
	uint8x16_t se_element;
	uint8x16_t min_d;

	//ブロックのための領域確保
	uint8_t *block;
	block = (uint8_t *)alloca( sizeof(uint8_t) * NS5 * (NC5 + W + NC5) );

	for(y=0; y<NC5; y++) //ブロックの初期化，0から第NC-1ラインまで
	{
		for(x=0; x < 2*NC5+W; x++)
		{
			block[(2*NC5+W) * y + x] = 0;
		}
	}
	for(y=NC5; y<NS5; y++) //ブロックの初期化，第NCラインから第2NC-1ラインまで
	{
		for(x=   0; x <      NC5; x++) block[(2*NC5+W) * y + x] = 0;
		for(x=  NC5; x <   NC5+W; x++) block[(2*NC5+W) * y + x] = input_data[(y-NC5) * W + (x - NC5)];
		for(x=NC5+W; x < 2*NC5+W; x++) block[(2*NC5+W) * y + x] = 0;
	}

	for(y=0; y < H; y++)
	{
        for(x=0; x < W; x = x+16)
		{
			//ブロックからpixel8への展開を書く
			for(xs=-NC5; xs<=NC5; xs++)
			{
				for(ys=-NC5; ys<=NC5; ys++)
				{
					pixel8[NS5 *(ys+NC5) + (xs+NC5)] = vld1q_u8(&(block[(2*NC5+W) * (NC5+ys) + (x+xs+NC5)]));
				}
			}

			min_d = vdupq_n_u8(255); //出力を最大輝度255に
			for(j=0; j<K; j++)
			{
				temp_max =veorq_u8(temp_max, temp_max); //各ビットでexorをとって０にクリア
				for(sx=0; sx<NS5*NS5; sx=sx+1)
				{
					//se_element = vdup_n_u8(se[NS5*NS5*j + sx]); //構造要素の一つの値をレジスタのすべてのレーンへコピー
					se_element = vdupq_n_u8(se[sx * K + j]);
					pixel = vqsubq_u8(pixel8[sx], se_element); //飽和演算で引き算，負の値は自動的に0に
					temp_max = vmaxq_u8(temp_max, pixel); //現在の最大値と比較，各レーンで大きい値が出力
				}
				min_d = vminq_u8(min_d, temp_max); //ダイレーションの結果の最小値を導出
			}
			//出力の書き込み先の変更
			vst1q_u8(&(out_data[y * W + x]), min_d); //出力画像のアドレスへストア
		}

		//ブロックの更新
		for(ys=0; ys<NS5-1; ys++)
		{
			for(xs=0; xs<2*NC5+W; xs++)
			{
				block[(2*NC5+W)*ys + xs] = block[(2*NC5+W)*(ys+1) + xs];
			}
		}
		if((y + NC5) < H)
		{
			for(xs=   0; xs <      NC5; xs++) block[(2*NC5+W)*(NS5 - 1) + xs] = 0;
			for(xs=  NC5; xs <   NC5+W; xs++) block[(2*NC5+W)*(NS5 - 1) + xs] = input_data[W * (y + NC5 + 1) + xs - NC5];
			for(xs=NC5+W; xs < 2*NC5+W; xs++) block[(2*NC5+W)*(NS5 - 1) + xs] = 0;
		}
		else
		{
			for(xs=   0; xs < 2*NC5+W; xs++) block[(2*NC5+W)*(NS5 - 1) + xs] = 0;
		}
	}
}

void dilation_simd_7x7(uint8_t input_data[], uint8_t out_data[], uint8_t se[], int H, int W, int K)
{
	// H: 画像の垂直方向の画素数, W: 画像の水平方向への画素数, K: 構造要素の数
	int sx, sy;
	int i, j, x, y, xs, ys;
	uint8x16_t temp_max; //SIMD用のレジスタ
	uint8x16_t pixel8[NS7*NS7]; //まずは，メモリからローカルな変数へコピーするための配列
	uint8x16_t pixel;
	uint8x16_t se_element;
	uint8x16_t min_d;

	//ブロックのための領域確保
	uint8_t *block;
	block = (uint8_t *)alloca( sizeof(uint8_t) * NS7 * (NC7 + W + NC7) );

	for(y=0; y<NC7; y++) //ブロックの初期化，0から第NC-1ラインまで
	{
		for(x=0; x < 2*NC7+W; x++)
		{
			block[(2*NC7+W) * y + x] = 0;
		}
	}
	for(y=NC7; y<NS7; y++) //ブロックの初期化，第NCラインから第2NC-1ラインまで
	{
		for(x=    0; x <     NC7; x++) block[(2*NC7+W) * y + x] = 0;
		for(x=  NC7; x <   NC7+W; x++) block[(2*NC7+W) * y + x] = input_data[(y-NC7) * W + (x - NC7)];
		for(x=NC7+W; x < 2*NC7+W; x++) block[(2*NC7+W) * y + x] = 0;
	}

	for(y=0; y < H; y++)
	{
        for(x=0; x < W; x = x+16)
		{
			//ブロックからpixel8への展開を書く
			for(xs=-NC7; xs<=NC7; xs++)
			{
				for(ys=-NC7; ys<=NC7; ys++)
				{
					pixel8[NS7 *(ys+NC7) + (xs+NC7)] = vld1q_u8(&(block[(2*NC7+W) * (NC7+ys) + (x+xs+NC7)]));
				}
			}

			min_d = vdupq_n_u8(255); //出力を最大輝度255に
			for(j=0; j<K; j++)
			{
				temp_max =veorq_u8(temp_max, temp_max); //各ビットでexorをとって０にクリア
				for(sx=0; sx<NS7*NS7; sx=sx+1)
				{
					//se_element = vdup_n_u8(se[NS5*NS5*j + sx]); //構造要素の一つの値をレジスタのすべてのレーンへコピー
					se_element = vdupq_n_u8(se[sx * K + j]);
					pixel = vqsubq_u8(pixel8[sx], se_element); //飽和演算で引き算，負の値は自動的に0に
					temp_max = vmaxq_u8(temp_max, pixel); //現在の最大値と比較，各レーンで大きい値が出力
				}
				min_d = vminq_u8(min_d, temp_max); //ダイレーションの結果の最小値を導出
			}
			//出力の書き込み先の変更
			vst1q_u8(&(out_data[y * W + x]), min_d); //出力画像のアドレスへストア
		}

		//ブロックの更新
		for(ys=0; ys<NS7-1; ys++)
		{
			for(xs=0; xs<2*NC7+W; xs++)
			{
				block[(2*NC7+W)*ys + xs] = block[(2*NC7+W)*(ys+1) + xs];
			}
		}
		if((y + NC7) < H)
		{
			for(xs=    0; xs <     NC7; xs++) block[(2*NC7+W)*(NS7 - 1) + xs] = 0;
			for(xs=  NC7; xs <   NC7+W; xs++) block[(2*NC7+W)*(NS7 - 1) + xs] = input_data[W * (y + NC7 + 1) + xs - NC7];
			for(xs=NC7+W; xs < 2*NC7+W; xs++) block[(2*NC7+W)*(NS7 - 1) + xs] = 0;
		}
		else
		{
			for(xs=   0; xs < 2*NC7+W; xs++) block[(2*NC7+W)*(NS5 - 1) + xs] = 0;
		}
	}
}

void erosion_simd_5x5(uint8_t input_data[], uint8_t out_data[], uint8_t se[], int H, int W, int K)
{
    	// H: 画像の垂直方向の画素数, W: 画像の水平方向への画素数, K: 構造要素の数
	int sx, sy;
	int i, j, x, y, xs, ys;
	uint8x16_t temp_min; //SIMD用のレジスタ
	uint8x16_t pixel8[NS5*NS5]; //まずは，メモリからローカルな変数へコピーするための配列
	uint8x16_t pixel;
	uint8x16_t se_element;
	uint8x16_t max_e;

	//ブロックのための領域確保
	uint8_t *block;
	block = (uint8_t *)alloca( sizeof(uint8_t) * NS5 * (NC5 + W + NC5) );

	for(y=0; y<NC5; y++) //ブロックの初期化，0から第NC-1ラインまで
	{
		for(x=0; x < 2*NC5+W; x++)
		{
			block[(2*NC5+W) * y + x] = 255;
		}
	}
	for(y=NC5; y<NS5; y++) //ブロックの初期化，第NCラインから第2NC-1ラインまで
	{
		for(x=   0; x <      NC5; x++) block[(2*NC5+W) * y + x] = 255;
		for(x=  NC5; x <   NC5+W; x++) block[(2*NC5+W) * y + x] = input_data[(y-NC5) * W + (x - NC5)];
		for(x=NC5+W; x < 2*NC5+W; x++) block[(2*NC5+W) * y + x] = 255;
	}

	for(y=0; y<H; y++)
	{
		for(x=0; x<W; x=x+8)
		{
			//ブロックからpixel8への展開を書く
			for(xs=-NC5; xs<=NC5; xs++)
			{
				for(ys=-NC5; ys<=NC5; ys++)
				{
					//pixel8[NS5 *(-ys+NC5) + (-xs+NC5)] = vld1q_u8(&(block[(2*NC5+W) * (NC5+ys) + (x+xs+NC5)]));
					pixel8[NS5 *(ys+NC5) + (xs+NC5)] = vld1q_u8(&(block[(2*NC5+W) * (NC5+ys) + (x+xs+NC5)]));
				}
			}

			max_e = veorq_u8(max_e, max_e); //各ビットでexorをとって0にクリア
			for(j=0; j<K; j++)
			{
				temp_min = vdupq_n_u8(255); //各レーンで255を代入
				for(sx=0; sx<NS5*NS5; sx=sx+1)
				{
					se_element = vdupq_n_u8(se[(NS5*NS5-1-sx) * K + j]);
					pixel = vqaddq_u8(pixel8[sx], se_element); //飽和演算で加算，255を超えた値は自動的に0に
					temp_min = vminq_u8(temp_min, pixel); //現在の最大値と比較，各レーンで大きい値が出力
				}
				max_e = vmaxq_u8(max_e, temp_min); //エロージョンの結果の最小値を導出
			}
			//出力の書き込み先の変更
			vst1q_u8(&(out_data[y*W+x]), max_e); //出力画像のアドレスへストア
		}

		//ブロックの更新
		for(ys=0; ys<NS5-1; ys++)
		{
			for(xs=0; xs<2*NC5+W; xs++)
			{
				block[(2*NC5+W)*ys + xs] = block[(2*NC5+W)*(ys+1) + xs];
			}
		}
		if((y + NC5) < H)
		{
			for(xs=   0; xs <      NC5; xs++) block[(2*NC5+W)*(NS5 - 1) + xs] = 255;
			for(xs=  NC5; xs <   NC5+W; xs++) block[(2*NC5+W)*(NS5 - 1) + xs] = input_data[W * (y + NC5 + 1) + xs - NC5];
			for(xs=NC5+W; xs < 2*NC5+W; xs++) block[(2*NC5+W)*(NS5 - 1) + xs] = 255;
		}
		else
		{
			for(xs=   0; xs < 2*NC5+W; xs++) block[(2*NC5+W)*(NS5 - 1) + xs] = 0;
		}
	}
}

void erosion_simd_7x7(uint8_t input_data[], uint8_t out_data[], uint8_t se[], int H, int W, int K)
{
    	// H: 画像の垂直方向の画素数, W: 画像の水平方向への画素数, K: 構造要素の数
	int sx, sy;
	int i, j, x, y, xs, ys;
	uint8x16_t temp_min; //SIMD用のレジスタ
	uint8x16_t pixel8[NS7*NS7]; //まずは，メモリからローカルな変数へコピーするための配列
	uint8x16_t pixel;
	uint8x16_t se_element;
	uint8x16_t max_e;

	//ブロックのための領域確保
	uint8_t *block;
	block = (uint8_t *)alloca( sizeof(uint8_t) * NS7 * (NC7 + W + NC7) );


	for(y=0; y<NC7; y++) //ブロックの初期化，0から第NC-1ラインまで
	{
		for(x=0; x < 2*NC7+W; x++)
		{
			block[(2*NC7+W) * y + x] = 255;
		}
	}
	for(y=NC7; y<NS7; y++) //ブロックの初期化，第NCラインから第2NC-1ラインまで
	{
		for(x=    0; x <     NC7; x++) block[(2*NC7+W) * y + x] = 255;
		for(x=  NC7; x <   NC7+W; x++) block[(2*NC7+W) * y + x] = input_data[(y-NC7) * W + (x - NC7)];
		for(x=NC7+W; x < 2*NC7+W; x++) block[(2*NC7+W) * y + x] = 255;
	}

	for(y=0; y<H; y++)
	{
		for(x=0; x<W; x=x+16)
		{

			//ブロックからpixel8への展開を書く
			for(xs=-NC7; xs<=NC7; xs++)
			{
				for(ys=-NC7; ys<=NC7; ys++)
				{
					pixel8[NS7 *(ys+NC7) + (xs+NC7)] = vld1q_u8(&(block[(2*NC7+W) * (NC7+ys) + (x+xs+NC7)]));
				}

			}

			max_e = veorq_u8(max_e, max_e); //各ビットでexorをとって0にクリア
			for(j=0; j<K; j++)
			{
				temp_min = vdupq_n_u8(255); //各レーンで255を代入
				for(sx=0; sx<NS7*NS7; sx=sx+1)
				{
					se_element = vdupq_n_u8(se[(NS7*NS7-1-sx) * K + j]);
					pixel = vqaddq_u8(pixel8[sx], se_element); //飽和演算で加算，255を超えた値は自動的に255に
					temp_min = vminq_u8(temp_min, pixel); //現在の最大値と比較，各レーンで大きい値が出力
				}
				max_e = vmaxq_u8(max_e, temp_min); //エロージョンの結果の最小値を導出
			}
			//出力の書き込み先の変更
			vst1q_u8(&(out_data[y*W+x]), max_e); //出力画像のアドレスへストア
		}

		//ブロックの更新
		for(ys=0; ys<NS7-1; ys++)
		{
			for(xs=0; xs<2*NC7+W; xs++)
			{
				block[(2*NC7+W)*ys + xs] = block[(2*NC7+W)*(ys+1) + xs];
			}
		}
		if((y + NC7) < H)
		{
			for(xs=    0; xs <     NC7; xs++) block[(2*NC7+W)*(NS7 - 1) + xs] = 255;
			for(xs=  NC7; xs <   NC7+W; xs++) block[(2*NC7+W)*(NS7 - 1) + xs] = input_data[W * (y + NC7 + 1) + xs - NC7];
			for(xs=NC7+W; xs < 2*NC7+W; xs++) block[(2*NC7+W)*(NS7 - 1) + xs] = 255;
		}
		else
		{
			for(xs=   0; xs < 2*NC7+W; xs++) block[(2*NC7+W)*(NS7 - 1) + xs] = 255;
		}
	}
}

void ave(uint8_t input_data1[], uint8_t input_data2[], uint8_t out_data[], int H, int W)
{
	uint8x16_t data1, data2, out_ave;
	int x, y;

	for(y=0; y<H; y++)
	{
		for(x=0; x<W; x=x+16)
		{
			data1 = vld1q_u8(&input_data1[y*W + x]);
			data2 = vld1q_u8(&input_data2[y*W + x]);
			out_ave = vhaddq_u8(data1, data2);
			vst1q_u8(&(out_data[y*W+x]), out_ave);
     }
   }
 }

void laplacian_simd_7x7(uint8_t input_data[], uint8_t out_data[], uint8_t se[], int H, int W, int K)
{
    	// H: 画像の垂直方向の画素数, W: 画像の水平方向への画素数, K: 構造要素の数
	int sx, sy;
	int i, j, x, y, xs, ys;
	uint8x16_t temp_min; //SIMD用のレジスタ
	uint8x16_t temp_max;

	uint8x16_t pixel8[NS7*NS7]; //まずは，メモリからローカルな変数へコピーするための配列
	uint8x16_t pixel;
	uint8x16_t se_element;

	uint8x16_t max_e;
	uint8x16_t min_d;

	uint8x16_t out_ave;

	//ブロックのための領域確保
	uint8_t *block;
	block = (uint8_t *)alloca( sizeof(uint8_t) * NS7 * (NC7 + W + NC7) );


	for(y=0; y<NC7; y++) //ブロックの初期化，0から第NC-1ラインまで
	{
		for(x=0; x < 2*NC7+W; x++)
		{
			block[(2*NC7+W) * y + x] = 255;
		}
	}
	for(y=NC7; y<NS7; y++) //ブロックの初期化，第NCラインから第2NC-1ラインまで
	{
		for(x=    0; x <     NC7; x++) block[(2*NC7+W) * y + x] = 255;
		for(x=  NC7; x <   NC7+W; x++) block[(2*NC7+W) * y + x] = input_data[(y-NC7) * W + (x - NC7)];
		for(x=NC7+W; x < 2*NC7+W; x++) block[(2*NC7+W) * y + x] = 255;
	}

	for(y=0; y<H; y++)
	{
		for(x=0; x<W; x=x+16)
		{

			//ブロックからpixel8への展開を書く
			for(xs=-NC7; xs<=NC7; xs++)
			{
				for(ys=-NC7; ys<=NC7; ys++)
				{
					pixel8[NS7 *(ys+NC7) + (xs+NC7)] = vld1q_u8(&(block[(2*NC7+W) * (NC7+ys) + (x+xs+NC7)]));
				}

			}

			max_e = veorq_u8(max_e, max_e); //各ビットでexorをとって0にクリア
			min_d = vdupq_n_u8(255); //出力を最大輝度255に

			for(j=0; j<K; j++)
			{
				temp_min = vdupq_n_u8(255); //各レーンで255を代入
				temp_max =veorq_u8(temp_max, temp_max); //各ビットでexorをとって０にクリア
				for(sx=0; sx<NS7*NS7; sx=sx+1)
				{
					se_element = vdupq_n_u8(se[(NS7*NS7-1-sx) * K + j]);
					pixel = vqaddq_u8(pixel8[sx], se_element); //飽和演算で加算，255を超えた値は自動的に255に
					temp_min = vminq_u8(temp_min, pixel); //現在の最小値と比較，各レーンで小さい値が出力

					se_element = vdupq_n_u8(se[sx * K + j]);
					pixel = vqsubq_u8(pixel8[sx], se_element); //飽和演算で引き算，負の値は自動的に0に
					temp_max = vmaxq_u8(temp_max, pixel); //現在の最大値と比較，各レーンで大きい値が出力
				}
				max_e = vmaxq_u8(max_e, temp_min); //エロージョンの結果の最小値を導出
				min_d = vminq_u8(min_d, temp_max); //ダイレーションの結果の最小値を導出
			}
			out_ave = vhaddq_u8(max_e, min_d);
			//出力の書き込み先の変更
			vst1q_u8(&(out_data[y*W+x]), out_ave); //出力画像のアドレスへストア
		}

		//ブロックの更新
		for(ys=0; ys<NS7-1; ys++)
		{
			for(xs=0; xs<2*NC7+W; xs++)
			{
				block[(2*NC7+W)*ys + xs] = block[(2*NC7+W)*(ys+1) + xs];
			}
		}
		if((y + NC7) < H)
		{
			for(xs=    0; xs <     NC7; xs++) block[(2*NC7+W)*(NS7 - 1) + xs] = 255;
			for(xs=  NC7; xs <   NC7+W; xs++) block[(2*NC7+W)*(NS7 - 1) + xs] = input_data[W * (y + NC7 + 1) + xs - NC7];
			for(xs=NC7+W; xs < 2*NC7+W; xs++) block[(2*NC7+W)*(NS7 - 1) + xs] = 255;
		}
		else
		{
			for(xs=   0; xs < 2*NC7+W; xs++) block[(2*NC7+W)*(NS7 - 1) + xs] = 255;
		}
	}
}

void laplacian_simd_5x5(uint8_t input_data[], uint8_t out_data[], uint8_t se[], int H, int W, int K)
{
    	// H: 画像の垂直方向の画素数, W: 画像の水平方向への画素数, K: 構造要素の数
	int sx, sy;
	int i, j, x, y, xs, ys;
	uint8x16_t temp_min; //SIMD用のレジスタ
	uint8x16_t temp_max;

	uint8x16_t pixel8[NS5*NS5]; //まずは，メモリからローカルな変数へコピーするための配列
	uint8x16_t pixel;
	uint8x16_t se_element;

	uint8x16_t max_e;
	uint8x16_t min_d;

	uint8x16_t out_ave;

	//ブロックのための領域確保
	uint8_t *block;
	block = (uint8_t *)alloca( sizeof(uint8_t) * NS5 * (NC5 + W + NC5) );


	for(y=0; y<NC5; y++) //ブロックの初期化，0から第NC-1ラインまで
	{
		for(x=0; x < 2*NC5+W; x++)
		{
			block[(2*NC5+W) * y + x] = 255;
		}
	}
	for(y=NC5; y<NS5; y++) //ブロックの初期化，第NCラインから第2NC-1ラインまで
	{
		for(x=    0; x <     NC5; x++) block[(2*NC5+W) * y + x] = 255;
		for(x=  NC5; x <   NC5+W; x++) block[(2*NC5+W) * y + x] = input_data[(y-NC5) * W + (x - NC5)];
		for(x=NC5+W; x < 2*NC5+W; x++) block[(2*NC5+W) * y + x] = 255;
	}

	for(y=0; y<H; y++)
	{
		for(x=0; x<W; x=x+16)
		{

			//ブロックからpixel8への展開を書く
			for(xs=-NC5; xs<=NC5; xs++)
			{
				for(ys=-NC5; ys<=NC5; ys++)
				{
					pixel8[NS5 *(ys+NC5) + (xs+NC5)] = vld1q_u8(&(block[(2*NC5+W) * (NC5+ys) + (x+xs+NC5)]));
				}

			}

			max_e = veorq_u8(max_e, max_e); //各ビットでexorをとって0にクリア
			min_d = vdupq_n_u8(255); //出力を最大輝度255に

			for(j=0; j<K; j++)
			{
				temp_min = vdupq_n_u8(255); //各レーンで255を代入
				temp_max =veorq_u8(temp_max, temp_max); //各ビットでexorをとって０にクリア
				for(sx=0; sx<NS5*NS5; sx=sx+1)
				{
					se_element = vdupq_n_u8(se[(NS5*NS5-1-sx) * K + j]);
					pixel = vqaddq_u8(pixel8[sx], se_element); //飽和演算で加算，255を超えた値は自動的に255に
					temp_min = vminq_u8(temp_min, pixel); //現在の最小値と比較，各レーンで小さい値が出力

					se_element = vdupq_n_u8(se[sx * K + j]);
					pixel = vqsubq_u8(pixel8[sx], se_element); //飽和演算で引き算，負の値は自動的に0に
					temp_max = vmaxq_u8(temp_max, pixel); //現在の最大値と比較，各レーンで大きい値が出力
				}
				max_e = vmaxq_u8(max_e, temp_min); //エロージョンの結果の最小値を導出
				min_d = vminq_u8(min_d, temp_max); //ダイレーションの結果の最小値を導出
			}
			out_ave = vhaddq_u8(max_e, min_d);
			//出力の書き込み先の変更
			vst1q_u8(&(out_data[y*W+x]), out_ave); //出力画像のアドレスへストア
		}

		//ブロックの更新
		for(ys=0; ys<NS5-1; ys++)
		{
			for(xs=0; xs<2*NC5+W; xs++)
			{
				block[(2*NC5+W)*ys + xs] = block[(2*NC5+W)*(ys+1) + xs];
			}
		}
		if((y + NC5) < H)
		{
			for(xs=    0; xs <     NC5; xs++) block[(2*NC5+W)*(NS5 - 1) + xs] = 255;
			for(xs=  NC5; xs <   NC5+W; xs++) block[(2*NC5+W)*(NS5 - 1) + xs] = input_data[W * (y + NC5 + 1) + xs - NC5];
			for(xs=NC5+W; xs < 2*NC5+W; xs++) block[(2*NC5+W)*(NS5 - 1) + xs] = 255;
		}
		else
		{
			for(xs=   0; xs < 2*NC5+W; xs++) block[(2*NC5+W)*(NS5 - 1) + xs] = 255;
		}
	}
}
