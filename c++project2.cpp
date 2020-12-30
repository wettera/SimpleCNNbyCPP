#include <opencv2/opencv.hpp>
#include <iostream>
#include <math.h>
#include <iomanip>
#include <chrono>
#include "face_binary_cls.h"

using namespace cv;
using namespace std;

void matrix_product(fc_param fc, float* result, float* input) {

	for (int i = 0; i < fc.out_features; i++) {
		result[i] = 0;
		for (int j = 0; j < fc.in_features; j++) {
			result[i] += fc.p_weight[i*fc.in_features + j] * input[j];
		}
		result[i] += fc.p_bias[i];
	}
}

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

void maxpool2x2(float* result, float* input, int input_channel, int input_height, int input_width) {

	for (int o = 0; o < input_channel; o++) {
		for (int r = 0; r < input_height; r += 2) {
			for (int c = 0; c < input_width; c += 2) {
				result[o* (input_height / 2) * (input_width / 2) + (r / 2) * (input_width / 2) + c / 2] = maxpool_sliding(input, input_height, input_width, o, r, c);
			}
		}
	}

}

int main()
{
	auto start = std::chrono::steady_clock::now();
	Mat image = imread("bg00.jpg");
	float* input = new float[3 * image.rows * image.cols];

	//Normalization and change BGR to RGB
	for (int r = 0; r < image.rows; r++) {
		Vec3b* ptr = image.ptr<Vec3b>(r);
		for (int c = 0; c < image.cols; c++) {
			input[0 * (image.rows* image.cols) + r * image.cols + c] = float(ptr[c][2]) / 255.0f;
			input[1 * (image.rows* image.cols) + r * image.cols + c] = float(ptr[c][1]) / 255.0f;
			input[2 * (image.rows* image.cols) + r * image.cols + c] = float(ptr[c][0]) / 255.0f;
		}
	}


	conv_param conv0 = conv_params[0];

	// Padding input
	int input_padding_height = image.rows + 2;
	int input_padding_width = image.cols + 2;
	float* input_padding = new float[3 * input_padding_height * input_padding_width];

	for (int i = 0; i < 3 * input_padding_height * input_padding_width; i++) {
		input_padding[i] = conv0.pad;
	}
	for (int r = 0; r < image.rows; r++) {
		for (int c = 0; c < image.cols; c++) {
			input_padding[0 * input_padding_height * input_padding_width + (r + 1) *  input_padding_width + c + 1] = input[0 * (image.rows* image.cols) + r * image.cols + c];
			input_padding[1 * input_padding_height * input_padding_width + (r + 1) *  input_padding_width + c + 1] = input[1 * (image.rows* image.cols) + r * image.cols + c];
			input_padding[2 * input_padding_height * input_padding_width + (r + 1) *  input_padding_width + c + 1] = input[2 * (image.rows* image.cols) + r * image.cols + c];
		}
	}

	delete input;


	int result0_channel = conv0.out_channels;
	int result0_height = image.rows / 2;
	int result0_width = image.cols / 2;
	float* result0 = new float[result0_channel *  result0_height * result0_width];

	// 128 -> 64
	conv3x3(conv0, result0, result0_height, result0_width, input_padding, input_padding_height, input_padding_width);
    delete input_padding;

	int result0_pooled_channel = result0_channel;
	int result0_pooled_height = result0_height / 2;
	int result0_pooled_width = result0_width / 2;

	float* result0_pooled= new float[result0_pooled_channel * result0_pooled_height * result0_pooled_width];

	//64 -> 32
	maxpool2x2(result0_pooled, result0, result0_channel, result0_height, result0_width);
	delete result0;

	conv_param conv1 = conv_params[1];
	int result1_channel = conv1.out_channels;
	int result1_height = 30;
	int result1_width = 30;
	
	float* result1 = new float[result1_channel * result1_height * result1_width];

	//32 -> 30
	conv3x3(conv1, result1, result1_height, result1_width, result0_pooled, result0_pooled_height, result0_pooled_width);
	delete result0_pooled;

	int result1_pooled_channel = result1_channel;
	int result1_pooled_height = result1_height / 2;
	int result1_pooled_width = result1_width / 2;

	float* result1_pooled = new float[result1_pooled_channel * result1_pooled_height * result1_pooled_width];

	//30 -> 15
	maxpool2x2(result1_pooled, result1, result1_channel, result1_height, result1_width);
	delete result1;

	int input2_padding_channel = result1_pooled_channel;
	int input2_padding_height = result1_pooled_height + 2;
	int input2_padding_width = result1_pooled_width + 2;
	float* input2_padding = new float[input2_padding_channel * input2_padding_height * input2_padding_width];

	conv_param conv2 = conv_params[2];

	for (int i = 0; i < input2_padding_channel * input2_padding_height * input2_padding_width; i++) {
		input2_padding[i] = conv2.pad;
	}

	for (int i = 0; i < input2_padding_channel; i++) {
		for (int r = 0; r < result1_pooled_height; r++) {
			for (int c = 0; c < result1_pooled_width; c++) {
				input2_padding[i * input2_padding_height * input2_padding_width + (r + 1) *  input2_padding_width + c + 1] = result1_pooled[i * result1_pooled_height * result1_pooled_width + r * result1_pooled_width + c];
			}
		}
	}

	delete result1_pooled;

	int result2_channel = conv2.out_channels;
	int result2_height = 8;
	int result2_width = 8;

	float* result2 = new float[result2_channel * result2_height * result2_width];

	//15 -> 8
	conv3x3(conv2, result2, result2_height, result2_width, input2_padding, input2_padding_height, input2_padding_height);
	delete input2_padding;

	fc_param fc0 = fc_params[0];
	float* predict = new float[fc0.out_features];
	// FC
	matrix_product(fc0, predict, result2);
	delete result2;

	//softmax
	float sum = 0;
	for (int i = 0; i < fc0.out_features; i++) {
		sum += exp(predict[i]);
	}
	for (int i = 0; i < fc0.out_features; i++) {
		predict[i] = exp(predict[i]) / sum;
	}

	cout << "bg score: " << fixed << setprecision(6) << predict[0];
	cout << ", face score: " << fixed << setprecision(6) << predict[1] << endl;

	auto end = std::chrono::steady_clock::now();
	cout
		<< "Time consuming: "
		<< std::chrono::duration_cast<std::chrono::microseconds>(end -
			start).count() << " microseconds = "
		<< std::chrono::duration_cast<std::chrono::milliseconds>(end -
			start).count() << "ms"
		<< endl;

	return 0;
    
}
