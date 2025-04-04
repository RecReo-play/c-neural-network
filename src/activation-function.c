#include <activation-function.h>

data_type ReLU(data_type input) {
	return input > 0.0 ? input : 0.0;
}
data_type d_ReLU(data_type input) {
	return  input>0 ? 1 : 0;
}


data_type LReLU(data_type input) {
	return input > 0.0 ? input : 0.01*input;
}
data_type d_LReLU(data_type input) {
	return input>0 ? 1 : 0.01;
}


data_type sigmoid(data_type input) {
	return 1 / (1 + exp(0 - input));
}
data_type d_sigmoid(data_type input) {
	data_type sig = sigmoid(input);
	return sig * (1.0 - sig);
}


data_type d_tanh(data_type input) {
	data_type t = tanh(input);
	return 1.0 - t*t;
}
