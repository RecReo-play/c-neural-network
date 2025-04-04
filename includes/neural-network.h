#ifndef a0d6c1_NN
#define a0d6c1_NN

#include <stdio.h>
#include <string.h>
#include <stddef.h>
#include <stdarg.h>
#include <stdlib.h>
#include <stdint.h>
#include <float.h>
#include <endian.h>
#include <time.h>
#include <math.h>
#include <errno.h>

#include <linear-algebra.h>
#include <activation-function.h>
#include <term_colors.h>
#ifndef activation
//#define activation(x) ReLU(x)
#define activation(x) LReLU(x)
//#define activation(x) sigmoid(x)
#endif
#ifndef activation_derivative
//#define activation_derivative(x) d_ReLU(x)
#define activation_derivative(x) d_LReLU(x)
//#define activation_derivative(x) d_sigmoid(x)
#endif
#ifndef sfree
#define sfree(P) ({free(P);P=(void*)0;})
#endif
#ifndef data_type
#define data_type float
#define data_type_str #data_type
#endif

struct NN_layer {
	Matrix weights;
	Vector biases;
};

struct NeuralNetwork {
	uint16_t num_hidden_layers;
	uint32_t input_size;
	struct NN_layer* hidden_layers;
	struct NN_layer output_layer;
};

struct layer_vectors {
	Vector a;
	Vector z;
	Matrix weight_gradient;
	Vector bias_gradient;
};
struct layer_gradient {
	Matrix weight_gradient;
	Vector bias_gradient;
};


typedef short (*inputGenerator)(size_t index, Vector* dst);
typedef short (*labelGenerator)(size_t index, Vector* dst);
typedef struct {
	struct NeuralNetwork* NN;
	inputGenerator igen;
	labelGenerator lgen;
	size_t batch_start;
	size_t batch_size;
	struct layer_gradient* gradient;
	float* loss;
} NN_args;

short NN_layer_init(struct NN_layer* dst, uint32_t input_nodes, uint32_t nodes);
short NeuralNetwork_init(struct NeuralNetwork* dst, uint32_t input_size, uint16_t hidden_layers);
short NeuralNetwork_new(struct NeuralNetwork* dst, uint32_t input_size, uint16_t hidden_layers, ...);
void NeuralNetwork_free(struct NeuralNetwork* NN);

short NeuralNetwork_feed(struct NeuralNetwork* NN, Vector* input, Vector* dst);
short NeuralNetwork_train(NN_args args);
short NeuralNetwork_apply_gradient(struct NeuralNetwork* NeuralNetwork, struct layer_gradient* gradient, data_type lrate);

short NeuralNetwork_export(struct NeuralNetwork* NeuralNetwork, char* outputfile);
short NeuralNetwork_import(struct NeuralNetwork* NeuralNetwork, char* inputfile);

short gradient_to_file(struct NeuralNetwork* NeuralNetwork, struct layer_gradient* gradient, char* file);
short file_to_gradient(struct NeuralNetwork* NeuralNetwork, struct layer_gradient* gradient, char* file);

double NeuralNetwork_test(NN_args args);
#endif
