#include <linear-algebra.h>

short linear_err(char* msg, short code, char* func) {
	perror(func);
	puts(msg);
	return code;
}

short linear_death(char* msg, short code) {
	puts(msg);
	return code;
}

void vector_free(Vector* vector) {
	free(vector->V);
	vector->V = NULL;
}
void matrix_free(Matrix* matrix) {
	free(matrix->M);
	matrix->M = NULL;
}

data_type matrix_get(Matrix* m, uint32_t row, uint32_t column) {
	return *(m->M + row * m->columns + column);
}
void matrix_set(Matrix* m, uint32_t row, uint32_t column, data_type value) {
	*(m->M + row * m->columns + column) = value;
}
void matrix_add(Matrix* m, uint32_t row, uint32_t column, data_type value) {
	*(m->M + row * m->columns + column) += value;
}

data_type vector_get(Vector* m, uint32_t index) {
	return *(m->V + index);
}
void vector_set(Vector* m, uint32_t index, data_type value) {
	*(m->V + index) = value;
}
void vector_add(Vector* m, uint32_t index, data_type value) {
	*(m->V + index) += value;
}



short scale_v(Vector* v, data_type scalar) {
#ifndef NO_LINEAR_CHECKS
	if (!v)
		return linear_death("vector scale: Invalid vector argument = NULL", 1);
#endif
	for (uint32_t i = 0; i < v->size; i++)
		v->V[i] *= scalar;
	return 0;
}


data_type vector_sqrd_mod(Vector* vector) {
	data_type mod = 0.0;
	for (uint32_t i = 0; i<vector->size; i++)
		mod += vector->V[i]*vector->V[i];
	return mod;
}

data_type vector_mod(Vector* vector) {
	return sqrtf(vector_sqrd_mod(vector));
}

// TODO msg
short to_vector(Matrix* matrix, Vector* dst) {
#ifndef NO_LINEAR_CHECKS
	if (!matrix)
		return linear_death("to_vector: Invalid matrix argument (NULL)", 2);
	if (!dst)
		return linear_death("to_vector: Invalid destination argument (NULL)", 3);
	if (matrix->columns > 1)
		return linear_death("to_vector: Invalid matrix size (columns=0)", 4);
#endif
	dst->size = matrix->rows;
	dst->V = matrix->M;
	return 0;
}
short to_matrix(Vector* vector, Matrix* dst) {
#ifndef NO_LINEAR_CHECKS
	if (!vector)
		return linear_death("to_matrix: Invalid vector argument (NULL)", 2);
	if (!dst)
		return linear_death("to_matrix: Invalid destination argument (NULL)", 3);
#endif
	dst->columns = 1;
	dst->rows = vector->size;
	dst->M = vector->V;
	return 0;
}








short vector_new(Vector* dst, uint32_t size) {
	free(dst->V);
	return vector_init(dst, size);
}
short matrix_new(Matrix* dst, uint32_t rows, uint32_t columns) {
	free(dst->M);
	return matrix_init(dst, rows, columns);
}

short matrix_init(Matrix* dst, uint32_t rows, uint32_t columns) {
#ifndef NO_LINEAR_CHECKS
	if (!dst)
		return linear_death("matrix_init: vector argument = NULL", 2);
	if (!rows)
		return linear_death("matrix_init: rows = 0", 3);
	if (!columns)
		return linear_death("matrix_init: columns = 0", 3);
#endif
	uint32_t al = sizeof(data_type) * rows * columns;
	if (!(dst->M = malloc(al)))
		return linear_err("matrix_init: Failed to allocate memory", 1, "malloc");
	dst->columns = columns;
	dst->rows = rows;
	return 0;
}

short vector_init(Vector* dst, uint32_t size) {
#ifndef NO_LINEAR_CHECKS
	if (!dst)
		return linear_death("vector_init: Invalid vector argument (zero)", 2);
	if (!size)
		return linear_death("vector_init: Invalid size argument (zero)", 3);
#endif
	uint32_t al = sizeof(data_type) * size;
	if ( !(dst->V = malloc(al)))
		return linear_death("vector_init: Failed to allocate memory", 1);
	dst->size = size;
	return 0;
}






short add_mm(Matrix* M1, Matrix* M2, Matrix* sum) {
#ifndef NO_LINEAR_CHECKS
	if (!M1 || !M2 || !sum) return 11;
	if (M1->rows != M2->rows || M1->columns != M2->columns)
		return 1;
#endif
	uint32_t rows = M1->rows;
	uint32_t columns = M1->columns;
	for (uint32_t row = 0; row < rows; row++)
		for (uint32_t column = 0; column < columns; column++)
			matrix_set(sum, row, column, matrix_get(M1,row,column) + matrix_get(M2,row,column));
	return 0;
}

short add_vv_new(Vector* v1, Vector* v2, Vector* sum) {
	if (vector_new(sum, v1->size))
		return 1;
	if (add_vv(v1, v2, sum))
		vector_free(sum);
	else return 0;
	return 1;
}

short add_vv(Vector* v1, Vector* v2, Vector* sum) {
#ifndef NO_LINEAR_CHECKS
	if (!v1)
		return linear_death("add_vv: Invalid v1 argument (zero)", 2);
	if (!v2)
		return linear_death("add_vv: Invalid v2 argument (zero)", 3);
	if (!sum) sum = v1;
	if ((sum->size = v1->size) != v2->size) return 1;
#endif
	for (uint32_t i = 0; i < sum->size; i++)
		sum->V[i] = v1->V[i] + v2->V[i];
	return 0;
}

short sub_vv_new(Vector* v1, Vector* v2, Vector* sum) {
	if (vector_new(sum, v1->size))
		return 1;
	if (sub_vv(v1, v2, sum))
		vector_free(sum);
	else return 0;
	return 1;
}

short sub_vv(Vector* v1, Vector* v2, Vector* sum) {
#ifndef NO_LINEAR_CHECKS
	if (!v1)
		return linear_death("sub_vv: Invalid v1 argument (zero)", 2);
	if (!v2)
		return linear_death("sub_vv: Invalid v2 argument (zero)", 3);
	if (!sum) sum = v1;
	if ((sum->size = v1->size) != v2->size)
		return linear_death("sub_vv: Vector sizes don't match", 1);
#endif
	uint32_t size = v1->size;
	for (uint32_t i = 0; i < size; i++)
		sum->V[i] = v1->V[i] - v2->V[i];
	return 0;
}

short multiply_mm_new(Matrix* M1, Matrix* M2, Matrix* dst) {
	if (matrix_new(dst, M1->rows, M2->columns))
		return 1;
	if (multiply_mm(M1, M2, dst))
		matrix_free(dst);
	else return 0;
	return 1;
}

short multiply_mm(Matrix* M1, Matrix* M2, Matrix* dst) {
#ifndef NO_LINEAR_CHECKS
	if (!M1 || !M2 || !dst) return 11;
	if (M1->columns != M2->rows) return 1;
#endif
	uint32_t m = M1->columns;
	uint32_t rows = M1->rows;
	uint32_t columns = M2->columns;
	dst->rows = rows;
	dst->columns = columns;
	for (uint32_t row = 0; row < rows; row++)
		for (uint32_t column = 0; column < columns; column++) {
			uint32_t n = row*columns + column;
			dst->M[n] = 0.0;
			for (uint32_t i = 0; i < m; i++)
				dst->M[n] += matrix_get(M1,row,column+i) * matrix_get(M2,row+i,column);
		}
	return 0;
}

short multiply_mv_new(Matrix* M, Vector* v, Vector* dst) {
	if (vector_new(dst, M->rows))
		return 1;
	if (multiply_mv(M, v, dst))
		vector_free(dst);
	else return 0;
	return 1;
}

short multiply_mv(Matrix* M, Vector* v, Vector* dst) {
#ifndef NO_LINEAR_CHECKS
	if (!M || !v || !dst) return 11;
	if (M->columns != v->size) return 1;
#endif
	uint32_t rows = M->rows;
	uint32_t size = M->columns;
	dst->size = rows;
	for (uint32_t row = 0; row < rows; row++) {
		dst->V[row] = 0.0;
		for (uint32_t i = 0; i < size; i++)
			dst->V[row] += matrix_get(M,row,i) * v->V[i];
	}
	return 0;
}


void vector_print(Vector vector) {
	puts(" _        _");
	for (uint32_t i = 0; i < vector.size; i++)
		printf("|%10.5f|\n", vector.V[i]);
	puts("|_        _|");
}















short add_vv2(Vector* v1, Vector* v2, Vector* sum) {
	if (!v1 || !v2 || !sum) return 11;
	if (v1->size != v2->size) return 1;
	Matrix M1;
	Matrix M2;
	Matrix Msum;
	short c;
	if (to_matrix(v1, &M1)) return 1;
	if (to_matrix(v1, &M2)) return 1;
	if (to_matrix(sum, &Msum)) return 1;
	if ((c = add_mm(&M1,&M2,&Msum))) return c;
	if (to_vector(&Msum, sum)) return 1;
	return 0;
}

short multiply_mv2(Matrix* M, Vector* v, Vector* dst) {
	if (!M || !v || !dst) return 11;
	if (M->columns != v->size) return 1;
	Matrix M2;
	Matrix product;
	short c;
	if (to_matrix(v, &M2)) return 1;
	if (to_matrix(dst, &product)) return 1;
	if ((c = multiply_mm(M,&M2,&product))) return c;
	if (to_vector(&product, dst)) return 1;
	return 0;
}
