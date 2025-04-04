#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
//#define NO_LINEAR_CHECKS
#include <neural-network.h>
#include <hash-table.h>
#include <errno.h>
#include <signal.h>

struct NeuralNetwork network;
#define TRAIN_DATASET_SIZE 300
#define TEST_DATASET_SIZE 200
#define BATCH_SIZE 300
float** train_input;
float train_output[TRAIN_DATASET_SIZE];
float** test_input;
float test_output[TEST_DATASET_SIZE];

size_t pseudorand(size_t* seed) {
	// constants for the lcg
	const size_t a = 1664525; // multiplier
	const size_t c = 1013904223; // increment
	const size_t m = 4294967296; // 2^32

	// update the seed and return the next random number
	*seed= (*seed * a + c) % m;
	return *seed % 255;
}

short trainDataGen(size_t index, Vector* dst) {
	dst->V[0] = train_input[index][0]; 
	dst->V[1] = train_input[index][1]; 
	return 0;
}

short trainLabelGen(size_t index, Vector* dst) {
	dst->V[0] = train_output[index];
	return 0;
}

short testDataGen(size_t index, Vector* dst) {
	dst->V[0] = test_input[index][0]; 
	dst->V[1] = test_input[index][1]; 
	return 0;
}

short testLabelGen(size_t index, Vector* dst) {
	dst->V[0] = test_output[index];
	return 0;
}

void new() {
	if (!NeuralNetwork_new(&network, 2, 2, 4, 2, 1))
		return;
	puts("failed to create a neural network");
	exit(103);
}

int die(int err, char* msg) {
	fprintf(stderr, msg);
	exit(err);
}

char arrcmp(data_type* a, data_type* b, uint32_t size) {
	for (uint32_t i = 0; i < size; i++)
		if (a[i] != b[i])
			return 0;
	return 1;
}

void generate_circular_data(float** input, float* output, size_t size) {
	float radius, angle;
	for (size_t i = 0; i < size; i++) {
		radius = (float)rand() / RAND_MAX; // Random radius in [0, 1]
		if (0.25 < radius && radius < 0.4) {
			i--;
			continue;
		}
		angle = (float)rand() / RAND_MAX * 2 * M_PI; // Random angle
		input[i][0] = radius * cos(angle);
		input[i][1] = radius * sin(angle);
		output[i] = (radius < 0.3) ? 0 : 1; // Classify based on radius
	}
}

FILE* graph;
FILE* gnuplot;

int main(int argc, char** argv) {

	struct layer_gradient gradient[3];
	float train_loss;
	float test_loss;
	int err = 0;
	char failed = 1;

	new();
	train_input = calloc(TRAIN_DATASET_SIZE, sizeof(float*));
	for (size_t i = 0; i < TRAIN_DATASET_SIZE; i++)
		train_input[i] = calloc(2, sizeof(float));
	test_input = calloc(TEST_DATASET_SIZE, sizeof(float*));
	for (size_t i = 0; i < TEST_DATASET_SIZE; i++)
		test_input[i] = calloc(2, sizeof(float));
	generate_circular_data(train_input, train_output, TRAIN_DATASET_SIZE);
	generate_circular_data(test_input, test_output, TEST_DATASET_SIZE);

	remove("graph");
	if (!(graph = fopen("graph", "w")))
		goto gfd_err;
	if (!(gnuplot = popen("gnuplot -persist", "w")))
		goto gnuplot_err;

	if (argc > 1 && !strcmp(argv[1], "-v")) {
		fprintf(gnuplot, "set terminal qt 2\n");
		fprintf(gnuplot, "set title 'Training set'\n");
		fprintf(gnuplot, "unset key\n");
		fprintf(gnuplot, "plot '-' using 1:2:($3) w p pt 7 lc variable\n");
		for (size_t i = 0; i < TRAIN_DATASET_SIZE; i++) {
			fprintf(gnuplot, "%f %f %d\n", train_input[i][0], train_input[i][1], train_output[i] ? 7 : 2);
		}
		fprintf(gnuplot, "e\n");
		fprintf(gnuplot, "set terminal qt 3\n");
		fprintf(gnuplot, "set title 'Testing set'\n");
		fprintf(gnuplot, "plot '-' using 1:2:($3) w p pt 7 lc variable\n");
		for (size_t i = 0; i < TEST_DATASET_SIZE; i++) {
			fprintf(gnuplot, "%f %f %d\n", test_input[i][0], test_input[i][1], test_output[i] ? 7 : 2);
		}
		fprintf(gnuplot, "e\n");
	}

	fprintf(gnuplot, "set key\n");
	fprintf(gnuplot, "set terminal qt 1\n");
	fprintf(gnuplot, "set title 'Neural Network Training'\n");
	fprintf(gnuplot, "set xlabel 'Epoch'\n");
	fprintf(gnuplot, "set ylabel 'Loss'\n");
	fflush(gnuplot);

	sleep(1);
	time_t last_replot = time(NULL);
	time_t current_time = time(NULL);
	short n_batches = TRAIN_DATASET_SIZE / BATCH_SIZE;
	short batch;
	uint32_t batch_start;
	char plotted = 0;
	float learning_rate = 0.01f;
	for (int i = 0; ; i++) {
//		generate_circular_data(train_input, train_output, TRAIN_DATASET_SIZE);
//		generate_circular_data(test_input, test_output, TEST_DATASET_SIZE);
		batch = i % n_batches;
		batch_start = batch * BATCH_SIZE;
		if (NeuralNetwork_train((NN_args) {
					.NN = &network,
					.igen = trainDataGen,
					.lgen = trainLabelGen,
					.batch_start = batch_start,
					.batch_size = BATCH_SIZE,
					.gradient = gradient,
					.loss = &train_loss
				}))
			continue;
		NeuralNetwork_apply_gradient(&network, gradient, learning_rate);
//		generate_circular_data();
		NeuralNetwork_test((NN_args) {
				.NN = &network,
				.igen = &testDataGen,
				.lgen = &testLabelGen,
				.batch_start = 0,
				.batch_size = TEST_DATASET_SIZE,
				.loss = &test_loss
			});
		fprintf(graph, "%d\t%lf\t%lf\n", i, train_loss, test_loss);
		fflush(graph);
		current_time = time(NULL);
		if (i % 1000 == 0) {
			learning_rate -= 0.000001f;
			if (learning_rate < 0.000001f)
				learning_rate = 0.000001f;
		}
		if (current_time - last_replot > 1) {
			if (plotted)
				fwrite("replot\n", 1, 7, gnuplot);
			else
				fprintf(gnuplot, "plot 'graph' u 1:2 w l t 'Training loss',"
									  "'graph' u 1:3 w l t 'Test loss'\n");
			plotted = 1;
			fflush(gnuplot);
			last_replot = time(NULL);
		}
	}

	NeuralNetwork_free(&network);

	failed = 0;

	pclose(gnuplot);
	gnuplot_err: err++;
	fclose(graph);
	gfd_err:;

	char* msg[] = {
		"fd",
		"gnuplot"
	};

	if (failed)
		perror(msg[err]);
	return 0;
}
