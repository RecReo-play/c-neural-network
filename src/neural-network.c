#include <neural-network.h>

static data_type noNANs(data_type x) {
	return isnan(x) ? 0.0f : x;
}

// Function to generate a normally distributed random number using Box-Muller transform
data_type He_Init(float stddev) {
	float u1 = (float)rand() / RAND_MAX; // Uniform random number in (0,1)
	float u2 = (float)rand() / RAND_MAX; // Uniform random number in (0,1)
	float z0 = noNANs(sqrt(-2.0f * log(u1))) * cos(2.0f * M_PI * u2); // Standard normal random variable
	return z0 * stddev; // Scale and shift to desired mean and stddev
}

static short apply_activation(Vector* vector, Vector* dst) {
	if (!vector) return 1;

	if (!dst) dst = vector;
	else dst->size = vector->size;

	for (uint32_t i = 0; i < vector->size; i++)
		dst->V[i] = activation(vector->V[i]);
	return 0;
}

short NN_layer_init(struct NN_layer* dst, uint32_t input_nodes, uint32_t nodes) {
	if (!dst) return 11;
	if (matrix_init(&dst->weights, nodes, input_nodes))
		return 1;
	if (vector_init(&dst->biases, nodes))
		return 1;
	return 0;
}

uint32_t get_biggest_layer(struct NeuralNetwork* NN) {
	uint32_t max = NN->input_size;
	for (uint32_t i = 0; i < NN->num_hidden_layers; i++) 
		if (NN->hidden_layers[i].biases.size > max) max = NN->hidden_layers[i].biases.size;
	return NN->output_layer.biases.size > max ? NN->output_layer.biases.size : max;
}

short NeuralNetwork_init(struct NeuralNetwork* dst, uint32_t input_size, uint16_t hidden_layers) {
	dst->input_size = input_size;
	dst->num_hidden_layers = hidden_layers;
	if ( !(dst->hidden_layers = malloc(sizeof(struct NN_layer) * hidden_layers)) )
		return 1;
	return 0;
}

short NeuralNetwork_new(struct NeuralNetwork* dst, uint32_t input_size, uint16_t hidden_layers, ...) {
	NeuralNetwork_init(dst, input_size, hidden_layers);
	va_list args;
	va_start(args, hidden_layers);
	srand( (unsigned int) time(NULL));
	uint32_t prev_neurons = input_size;
	for (uint32_t i = 0; i <= hidden_layers; i++) {
		uint32_t neurons = va_arg(args, uint32_t);
		struct NN_layer* layer = i == hidden_layers ? &dst->output_layer : &dst->hidden_layers[i];
		if ( NN_layer_init(layer, prev_neurons, neurons))
			return 1;
		uint32_t values = prev_neurons * neurons;
		float stddev = sqrt(2.0f / prev_neurons); // He initialization standard deviation
		// Initialize weights using He initialization
		for (uint32_t j = 0; j < values; j++)
			layer->weights.M[j] = He_Init(stddev);
		// Initialize biases to zero (or you can use a small constant)
		for (uint32_t j = 0; j < neurons; j++)
			layer->biases.V[j] = 0.0f; // or nrand(0.0f, stddev) if you prefer

		prev_neurons = neurons;
	}
	va_end(args);
	return 0;
}

short NeuralNetwork_feed(struct NeuralNetwork* NN, Vector* input, Vector* dst) {
	int err = 0;
	char failed = 1;
	if (input->size != NN->input_size) goto INVALID_ARG_err;
	uint32_t max_size = get_biggest_layer(NN);
	Vector layer_input = {.size = input->size, .V = malloc(max_size * sizeof(data_type))};
	if (!layer_input.V) goto INPUT_ALLOC_err;
	Vector layer_output = {.size = 0, .V = malloc(max_size * sizeof(data_type))};
	if (!layer_output.V) goto OUTPUT_ALLOC_err;
	memcpy(layer_input.V, input->V, input->size * sizeof(data_type));
	struct NN_layer* layer;
	for (uint32_t i = 0; i < NN->num_hidden_layers; i++) {
		layer = &NN->hidden_layers[i];
		if (multiply_mv(&layer->weights, &layer_input, &layer_output))
			printf(FG_GRAY "[Neural Network] " C_RESET FG_RED FG_BRIGHT "Error applying multiplying weight matrix:" C_RESET " layer=%u\n", i);
		if (add_vv(&layer_output, &layer->biases, &layer_input))
			printf(FG_GRAY "[Neural Network] " C_RESET FG_RED FG_BRIGHT "Error adding bias:" C_RESET " layer=%u\n", i);
		if (apply_activation(&layer_input, NULL))
			printf(FG_GRAY "[Neural Network] " C_RESET FG_RED FG_BRIGHT "Error applying activation function:" C_RESET " layer=%u\n", i);
	}
	layer = &NN->output_layer;
	if (vector_init(dst, layer->biases.size)) goto DST_ALLOC_err;
	if (multiply_mv(&layer->weights, &layer_input, &layer_output))
		puts(FG_GRAY "[Neural Network] " C_RESET FG_RED FG_BRIGHT "Error multiplying weight matrix:" C_RESET " layer=output");
	if (add_vv(&layer_output, &layer->biases, dst))
		puts(FG_GRAY "[Neural Network] " C_RESET FG_RED FG_BRIGHT "Error adding bias:" C_RESET " layer=output");
	if (apply_activation(dst, NULL))
		puts(FG_GRAY "[Neural Network] " C_RESET FG_RED FG_BRIGHT "Error applying activation function:" C_RESET " layer=output");
	
	failed = 0;
DST_ALLOC_err: err++;
	vector_free(&layer_input);
OUTPUT_ALLOC_err: err++;
	vector_free(&layer_output);
INPUT_ALLOC_err: err++;
INVALID_ARG_err: err++;

	char* msg[] = {
		NULL,
		"Invalid arguments",
		"Failed to allocate memory for input vector",
		"Failed to allocate memory for output vector",
		"Failed to allocate memory for destination vector",
	};
	if (failed)
		printf(FG_GRAY "[Neural Network]" C_RESET FG_BRIGHT FG_RED "%s" C_RESET "\n", msg[err]);
	
	return failed ? err : 0;
}

void NN_layer_free(struct NN_layer layer) {
	matrix_free(&layer.weights);
	vector_free(&layer.biases);
}


void NeuralNetwork_free(struct NeuralNetwork* NN) {
	for (uint32_t i = 0; i < NN->num_hidden_layers; i++)
		NN_layer_free(NN->hidden_layers[i]);
	NN_layer_free(NN->output_layer);
	sfree(NN->hidden_layers);
}


short NeuralNetwork_calculate(struct NeuralNetwork* NN,	struct layer_vectors* lv) {

	if (!NN || !lv) return 11;

	struct NN_layer* layer;
	uint32_t pl = 0;	// previous layer
	register uint32_t i;
	uint32_t n = NN->num_hidden_layers + 1;

	for (i = 1; i < n; i++) {
		layer = &NN->hidden_layers[pl];

		if (multiply_mv(&layer->weights, &lv[pl].a, &lv[i].z))
			printf(FG_GRAY "[Neural Network Learning] " C_RESET FG_RED FG_BRIGHT "Error applying multiplying weight matrix:" C_RESET " layer=%u\n", i);
		if (add_vv(&lv[i].z, &layer->biases, &lv[i].z))
			printf(FG_GRAY "[Neural Network Learning] " C_RESET FG_RED FG_BRIGHT "Error adding bias:" C_RESET " layer=%u\n", i);
		if (apply_activation(&lv[i].z, &lv[i].a))
			printf(FG_GRAY "[Neural Network Learning] " C_RESET FG_RED FG_BRIGHT "Error applying activation function:" C_RESET " layer=%u\n", i);

		pl = i;
	}

	layer = &NN->output_layer;

	if (multiply_mv(&layer->weights, &lv[pl].a, &lv[i].z))
		puts(FG_GRAY "[Neural Network Learning] " C_RESET FG_RED FG_BRIGHT "Error multiplying weight matrix:" C_RESET " layer=output");
	if (add_vv(&lv[i].z, &layer->biases, &lv[i].z))
		puts(FG_GRAY "[Neural Network Learning] " C_RESET FG_RED FG_BRIGHT "Error adding bias:" C_RESET " layer=output");
	if (apply_activation(&lv[i].z, &lv[i].a))
		puts(FG_GRAY "[Neural Network Learning] " C_RESET FG_RED FG_BRIGHT "Error applying activation function:" C_RESET " layer=output");

	return 0;
}





short NeuralNetwork_backpropagation(struct NeuralNetwork* NN, struct layer_vectors* lv, Vector* dCda, Vector* temp_dCda) {

	data_type dadz, dadz_dCda;
	uint32_t layer, neuron, weight;
	struct NN_layer* current_layer = &NN->output_layer;
	Vector* prev_activations;
	data_type* dp_temp;
	// loop variables
	uint32_t mi;	// matrix index (calculation optimised)


	layer = NN->num_hidden_layers+1;
	while (1) {
		prev_activations = &lv[layer-1].a; // prev layer
		mi = 0;
		for (neuron = 0; neuron < current_layer->biases.size; neuron++) {
			dadz = activation_derivative(lv[layer].z.V[neuron]); // current layer
			dadz_dCda = dadz*dCda->V[neuron];
			lv[layer].bias_gradient.V[neuron] += /* the derivative is 1 */dadz_dCda;
			for (weight = 0; weight < current_layer->weights.columns; weight++) {
											/* the derivative of z evaluated on the weight
											 * 			= activation of prev neuron		*/
				lv[layer].weight_gradient.M[mi] += (prev_activations->V[weight])*dadz_dCda;
											/* the derivative of z evaluated on the activation
											 * 			= weight 			*/
				temp_dCda->V[weight] += current_layer->weights.M[mi]*dadz_dCda;
				mi++;
			}
		}
		if (layer > 1) layer--;
		else break;
		current_layer = &NN->hidden_layers[layer-1];
		dp_temp = dCda->V;
		dCda->V = temp_dCda->V;
		temp_dCda->V = dp_temp;
		memset(dp_temp, 0, dCda->size * sizeof(data_type));
		dCda->size = temp_dCda->size;
		temp_dCda->size = current_layer->biases.size;
	}

	return 0;
}

short NeuralNetwork_train(NN_args args) {

	// arg check
	if (!args.NN || !args.igen || !args.lgen || !args.batch_size) return 11;
	// variables
	uint32_t allocated_layers = 0;
	uint32_t max_layer_size = get_biggest_layer(args.NN);
	uint32_t d_memset_size = max_layer_size * sizeof(data_type);
	uint32_t n = args.NN->num_hidden_layers + 1;
	int gerr = 0;
	char gfailed = 1;
	// vectors
	struct layer_vectors layer_vectors[args.NN->num_hidden_layers + 2];
	Vector dCda_T;
	Vector temp_dCda_T;
	Vector desired;
	// loop variables
	int err = 0;
	char failed = 1;
	size_t endI = args.batch_start + args.batch_size;

	float backup_loss = 0.0f; // loss variable to store the loss into if not given
							  // instead of checking for NULL every loop cycle

	// initialisation
	if (vector_init(&desired, args.NN->output_layer.biases.size))
		goto DES_VEC_INIT_err;
	if (vector_init(&dCda_T, max_layer_size))
		goto dCda_VEC_INIT_err;
	if (vector_init(&temp_dCda_T, max_layer_size))
		goto temp_dCda_VEC_INIT_err;
	memset(temp_dCda_T.V, 0, temp_dCda_T.size * sizeof(data_type));
	if (vector_init(&layer_vectors[0].a, args.NN->input_size))
		goto INPUT_VEC_INIT_err;
	for (allocated_layers = 1; allocated_layers <= n; allocated_layers++) {
		struct NN_layer* layer = (allocated_layers == n) ? &args.NN->output_layer : &args.NN->hidden_layers[allocated_layers-1];
		uint32_t size = layer->biases.size;
		if (vector_init(&layer_vectors[allocated_layers].a, size) ||
			vector_init(&layer_vectors[allocated_layers].z, size) ||
			matrix_init(&layer_vectors[allocated_layers].weight_gradient, layer->weights.rows, layer->weights.columns) ||
			vector_init(&layer_vectors[allocated_layers].bias_gradient, size)) {
				printf("allocated layers: %u\n", allocated_layers);
				goto VEC_INIT_err;
		}
		memset(layer_vectors[allocated_layers].weight_gradient.M, 0, layer->weights.rows*layer->weights.columns);
		memset(layer_vectors[allocated_layers].bias_gradient.V, 0, layer->biases.size);
	}
	allocated_layers--;

	if (!args.loss) args.loss = &backup_loss;
	*args.loss = 0.0f;
	// loop
	for (size_t example = args.batch_start; example < endI; example++) {
		if (args.igen(example, &layer_vectors[0].a)) goto INPUT_GEN_err;
		if (NeuralNetwork_calculate(args.NN, layer_vectors)) goto PRE_CALC_err;
		if (args.lgen(example, &desired)) goto LABEL_GEN_err;
		dCda_T.size = desired.size;
		if (sub_vv(&layer_vectors[n].a, &desired, &dCda_T)) goto COST_VEC_err;
		for (uint32_t i = 0; i<dCda_T.size; i++)
			*args.loss += dCda_T.V[i]*dCda_T.V[i]; // added directly; no need for sqrt() the sum; squered length
		if (scale_v(&dCda_T, (float)1/args.batch_size)) goto SCALE_err;
		memset(temp_dCda_T.V, 0, d_memset_size);
		if (NeuralNetwork_backpropagation(args.NN, layer_vectors, &dCda_T, &temp_dCda_T)) goto BACKPROPAGATION_err;

		continue;
		BACKPROPAGATION_err: err++;
		SCALE_err: err++;
		MOD_err: err++;
		COST_VEC_err: err++;
		LABEL_GEN_err: err++;
		PRE_CALC_err: err++;
		INPUT_GEN_err: err++;
		char* msg[] = {
			NULL,
			"Failed to generate input",
			"Failed to precalculate neural network state",
			"Failed to generate a label",
			"Failed to calculate the cost vector",
			"Failed to get vector modulo",
			"Failed to scale the cost vector",
			"Backpropagation failed",
		};
		printf(FG_GRAY "[Neural Network Learning] " C_RESET FG_GREEN FG_BRIGHT "Example %zu - " C_RESET FG_RED FG_BRIGHT "%s" C_RESET "\n", example, msg[err]);
		// do something TODO
		err = 0;
	}

	*args.loss /= 1.0f/2.0f * (float)args.batch_size;

	for (uint16_t l = 1; l <= n; l++) {
		struct layer_vectors* lv = &layer_vectors[l];
		uint32_t weights = lv->weight_gradient.columns * lv->weight_gradient.rows;
		for (uint32_t weight = 0; weight < weights; weight++)
			lv->weight_gradient.M[weight] /= args.batch_size;
		for (uint32_t neuron = 0; neuron < lv->bias_gradient.size; neuron++)
			lv->bias_gradient.V[neuron] /= args.batch_size;
		args.gradient[l-1].weight_gradient = lv->weight_gradient;
		args.gradient[l-1].bias_gradient = lv->bias_gradient;
	}

	gfailed = 0;

	VEC_INIT_err: gerr++;
	for (allocated_layers--; 0 < allocated_layers; allocated_layers--) {
		struct NN_layer* layer = (allocated_layers == n) ? &args.NN->output_layer : &args.NN->hidden_layers[allocated_layers];
		vector_free(&layer_vectors[allocated_layers].a);
		vector_free(&layer_vectors[allocated_layers].z);
//		matrix_free(&layer_vectors[allocated_layers].weight_gradient);
//		vector_free(&layer_vectors[allocated_layers].bias_gradient);
	}
	vector_free(&layer_vectors[0].a);
	INPUT_VEC_INIT_err: gerr++;
	vector_free(&temp_dCda_T);
	temp_dCda_VEC_INIT_err: gerr++;
	vector_free(&dCda_T);
	dCda_VEC_INIT_err: gerr++;
	vector_free(&desired);
	DES_VEC_INIT_err: gerr++;
	ARG_err: gerr++;
	
	char* gmsg[] = {
		NULL,
		"Invalid arguments",
//		"Failed to allocate memory for output array",
		"Failed to pre-initialise the desired output vector",
		"Failed to pre-initialise the derivative vector",
		"Failed to pre-initialise the temporal derivative vector",
		"Failed to pre-initialise the input vector",
		"Failed to pre-initialise the pre-calculation vectors",
	};
	
	if (gfailed) printf(FG_GRAY "[Neural Network Training] " C_RESET FG_RED FG_BRIGHT "%s" C_RESET "\n", gmsg[gerr]);
	return gfailed ? gerr : 0;
}

short NeuralNetwork_gradient_free(struct NeuralNetwork* NeuralNetwork, struct layer_gradient* gradient) {
	for (uint32_t i = 0; i <= NeuralNetwork->num_hidden_layers; i++) {
		matrix_free(&gradient[i].weight_gradient);
		vector_free(&gradient[i].bias_gradient);
	}
	return 0;
}

short NeuralNetwork_apply_gradient(struct NeuralNetwork* NeuralNetwork, struct layer_gradient* gradient, data_type lrate) {
	struct NN_layer* layer;
	for (uint32_t i = 0; i <= NeuralNetwork->num_hidden_layers; i++) {
		layer = (i == NeuralNetwork->num_hidden_layers) ? &NeuralNetwork->output_layer : &NeuralNetwork->hidden_layers[i];
		uint32_t weights = gradient[i].weight_gradient.columns * gradient[i].weight_gradient.rows;
		for (uint32_t weight = 0; weight < weights; weight++)
			layer->weights.M[weight] -= lrate * gradient[i].weight_gradient.M[weight];
		for (uint32_t neuron = 0; neuron < gradient[i].bias_gradient.size; neuron++)
			layer->biases.V[neuron] -= lrate * gradient[i].bias_gradient.V[neuron];
	}
	return 0;
}




double NeuralNetwork_test(NN_args args) {
	size_t endI = args.batch_start + args.batch_size;
	Vector input, output, desired;
	char gfailed = 1;
	int gerr = 0;
	int err = 0;
	float highest_value, desired_highest_value;
	uint32_t highest_answer, desired_highest_answer;
	size_t correct = 0;

	// loss calculation
	float backup_loss = 0.0f; // loss variable to store the loss into if not given
							  // instead of checking for NULL every loop cycle
	float diff; // for calculating loss without additional vector
	if (vector_init(&input, args.NN->input_size)) goto INPUT_VECTOR_INIT_err;
	if (vector_init(&output, args.NN->output_layer.biases.size)) goto OUTPUT_VECTOR_INIT_err;
	if (vector_init(&desired, args.NN->output_layer.biases.size)) goto DESIRED_VECTOR_INIT_err;
	if (!args.loss) args.loss = &backup_loss;
	*args.loss = 0.0f;
	for (size_t example = args.batch_start; example < endI; example++) {
		if (args.igen(example, &input)) goto INPUT_GEN_err;
		if (args.lgen(example, &desired)) goto LABEL_GEN_err;
		if (NeuralNetwork_feed(args.NN, &input, &output)) goto FEED_err;

		// calculating loss without additional vector
		for (uint32_t i = 0; i < output.size; i++) {
			diff = output.V[i] - desired.V[i];
			*args.loss += diff*diff; // added directly; no need for sqrt() the sum; squered length
		}

/*
		highest_value = FLT_MIN;
		desired_highest_value = FLT_MIN;
		highest_answer = -1;
		desired_highest_answer = -1;
		for (uint32_t i = 0; i < desired.size; i++)
			if (desired.V[i] > highest_value) {
				desired_highest_value = output.V[i];
				desired_highest_answer = i;
			}
		for (uint32_t i = 0; i < output.size; i++)
			if (output.V[i] > highest_value) {
				highest_value = output.V[i];
				highest_answer = i;
			}
		if (highest_answer != -1 && desired_highest_answer != -1)
			if (highest_answer == desired_highest_answer)
				correct++;
*/
		switch ((int)desired.V[0]) {
			case 0:
				if (output.V[0] < 0.5)
					correct++;
				break;
			case 1:
				if (output.V[0] > 0.5)
					correct++;
				break;
		}

		continue;
		FEED_err: err++;
		LABEL_GEN_err: err++;
		INPUT_GEN_err: err++;
		char* msg[] = {
			NULL,
			"Failed to generate input",
			"Failed to precalculate neural network state",
			"Failed to generate a label",
			"Failed to calculate the cost vector",
			"Failed to scale the cost vector",
			"Backpropagation failed",
		};
		printf(FG_GRAY "[Neural Network Testing] " C_RESET FG_GREEN FG_BRIGHT "Example %zu - " C_RESET FG_RED FG_BRIGHT "%s" C_RESET "\n", example, msg[err]);
		err = 0;
	}

	*args.loss /= 1.0f/2.0f * (float)args.batch_size;

	gfailed = 0;

	vector_free(&desired);
	DESIRED_VECTOR_INIT_err: gerr++;
	vector_free(&output);
	OUTPUT_VECTOR_INIT_err: gerr++;
	vector_free(&input);
	INPUT_VECTOR_INIT_err: gerr++;
	char* msg[] = {
		NULL,
		"Failed to initialise input vector",
		"Failed to initialise output vector",
		"Failed to initialise desired vector",
	};
	if (gfailed)
		printf(FG_GRAY "[Neural Network Testing] " C_RESET FG_RED FG_BRIGHT "%s" C_RESET "\n", msg[err]);
	else
		return (double) correct / (double) args.batch_size;
	return -1;
}
