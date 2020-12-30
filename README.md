# **CNN Forward for Binary Face Classification using C++**

**Name**: 郜晨阳

**SID**: 11710904

## Introduction

Course project for **CS205 C/C++ Program Design**, implement a convolutional neural network (CNN) using C/C++. The task is binary classification(face or background). More details can be find in the [project requirement](https://github.com/wettera/SimpleCNNbyCPP/blob/main/Project2-update20201217.pdf)

### Requirement

opencv

## Implementation

### 3*3 convolution operation

```c++
float kernel_sliding(float* kernel, float* input, int in_channels, int rows, int colums, int o, int i, int r, int c) {
	float result = 0;

	result += kernel[o*(in_channels * 3 * 3) + i * (3 * 3) + 0] * input[i*rows*colums + r*colums + c]
	 + kernel[o*(in_channels * 3 * 3) + i * (3 * 3) + 1] * input[i*rows*colums + r * colums + c+1]
	 + kernel[o*(in_channels * 3 * 3) + i * (3 * 3) + 2] * input[i*rows*colums + r * colums + c + 2]

	 + kernel[o*(in_channels * 3 * 3) + i * (3 * 3) + 3] * input[i*rows*colums + (r + 1) * colums + c]
	 + kernel[o*(in_channels * 3 * 3) + i * (3 * 3) + 4] * input[i*rows*colums + (r + 1) * colums + c+1]
	 + kernel[o*(in_channels * 3 * 3) + i * (3 * 3) + 5] * input[i*rows*colums + (r + 1) * colums + c+2]

	 + kernel[o*(in_channels * 3 * 3) + i * (3 * 3) + 6] * input[i*rows*colums + (r + 2) * colums + c]
	 + kernel[o*(in_channels * 3 * 3) + i * (3 * 3) + 7] * input[i*rows*colums + (r + 2) * colums + c+1]
	 + kernel[o*(in_channels * 3 * 3) + i * (3 * 3) + 8] * input[i*rows*colums + (r + 2) * colums + c+2];

	return result;

}


void conv3x3(conv_param conv, float* result, int result_height, int result_width,  float* input, int input_height, int input_width) {

	for (int i = 0; i < conv.out_channels* result_height * result_width; i++) {
		result[i] = 0;
	}

	for (int o = 0; o < conv.out_channels; o++) {
		for (int i = 0; i < conv.in_channels; i++) {

			for (int j = 0; j < input_height - 2; j += conv.stride) {
				for (int k = 0; k < input_width - 2; k += conv.stride) {
					result[o * result_height * result_width + (j / conv.stride) * result_width + k / conv.stride] += kernel_sliding(conv.p_weight, input, conv.in_channels, input_height, input_width, o, i, j, k);
				}
			}
		}

		for (int r = 0; r < result_height; r++) {
			for (int c = 0; c < result_width; c++) {

				result[o * result_height * result_width + r * result_width + c] += conv.p_bias[o];
				//relu
				if (result[o * result_height * result_width + r * result_width + c] < 0) {
					result[o * result_height * result_width + r * result_width + c] = 0;
				}
			}
		}
	}

}
```

### 2*2 max pooling

```c++
float maxpool_sliding(float* input, int input_heigth, int input_width, int o, int r, int c) {
	float num1 = input[o*input_heigth * input_width + r * input_width + c];
	float num2 = input[o*input_heigth * input_width + r * input_width + c+1];
	float num3 = input[o*input_heigth * input_width + (r+1) * input_width + c];
	float num4 = input[o*input_heigth * input_width + (r+1) * input_width + c+1];

	float max = 0;
	if (num1 > max) {
		max = num1;
	}

	if (num2 > max) {
		max = num2;
	}

	if (num3 > max) {
		max = num3;
	}

	if (num4 > max) {
		max = num4;
	}

	return max;

}

void maxpool2x2(float* result, float* input, int input_channel, int input_height, int input_width) {

	for (int o = 0; o < input_channel; o++) {
		for (int r = 0; r < input_height; r += 2) {
			for (int c = 0; c < input_width; c += 2) {
				result[o* (input_height / 2) * (input_width / 2) + (r / 2) * (input_width / 2) + c / 2] = maxpool_sliding(input, input_height, input_width, o, r, c);
			}
		}
	}

}
```



### Fully connected layer

```c++
void matrix_product(fc_param fc, float* result, float* input) {

	for (int i = 0; i < fc.out_features; i++) {
		result[i] = 0;
		for (int j = 0; j < fc.in_features; j++) {
			result[i] += fc.p_weight[i*fc.in_features + j] * input[j];
		}
		result[i] += fc.p_bias[i];
	}
}
```



一个效率更高的实现：

```c++
void matrix_product_improved(fc_param fc, float* result, float* input) {

	for (int i = 0; i < fc.out_features; i++) {
		result[i] = 0;

		int j = 0;
		while (j + 8 < fc.in_features) {
			result[i] += fc.p_weight[i*fc.in_features + j] * input[j]
				+ fc.p_weight[i*fc.in_features + j + 1] * input[j + 1]
				+ fc.p_weight[i*fc.in_features + j + 2] * input[j + 2]
				+ fc.p_weight[i*fc.in_features + j + 3] * input[j + 3]
				+ fc.p_weight[i*fc.in_features + j + 4] * input[j + 4]
				+ fc.p_weight[i*fc.in_features + j + 5] * input[j + 5]
				+ fc.p_weight[i*fc.in_features + j + 6] * input[j + 6]
				+ fc.p_weight[i*fc.in_features + j + 7] * input[j + 7];
			j += 8;
		}
		while (j < fc.in_features) {
			result[i] += fc.p_weight[i*fc.in_features + j] * input[j];
			j++;
		}

		result[i] += fc.p_bias[i];
	}
}
```



## Result

/samples/face.jpg:

![https://github.com/wettera/SimpleCNNbyCPP/blob/main/result/face1.png]()

/samples/bg.jpg:

![https://github.com/wettera/SimpleCNNbyCPP/blob/main/result/bg1.png]()



使用效率更高的Fully connected layer实现方式：

/samples/face.jpg:

![https://github.com/wettera/SimpleCNNbyCPP/blob/main/result/face2.png]()

/samples/bg.jpg:

![https://github.com/wettera/SimpleCNNbyCPP/blob/main/result/bg2.png]()



