#ifndef c5c693
#define c5c693

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>
#include <stddef.h>
#include <term_colors.h>
#include <math.h>

#define sfree(P) ({free(P);P=(void*)0;})
#ifndef data_type
#define data_type float
#define data_type_str #data_type
#endif

typedef struct {
	uint32_t columns;
	uint32_t rows;
	data_type* M;
} Matrix;

typedef struct {
	uint32_t size;
	data_type* V;
} Vector;


void vector_free(Vector* vector);
void matrix_free(Matrix* matrix);
short to_vector(Matrix* matrix, Vector* dst);
short to_matrix(Vector* vector, Matrix* dst);

data_type matrix_get(Matrix* m, uint32_t row, uint32_t column);
data_type vector_get(Vector* m, uint32_t index);
void matrix_set(Matrix* m, uint32_t row, uint32_t column, data_type value);
void matrix_add(Matrix* m, uint32_t row, uint32_t column, data_type value);
void vector_set(Vector* m, uint32_t index, data_type value);
void vector_add(Vector* m, uint32_t index, data_type value);

short vector_init(Vector* dst, uint32_t size);
short matrix_init(Matrix* dst, uint32_t rows, uint32_t columns);
short vector_new(Vector* dst, uint32_t size);
short matrix_new(Matrix* dst, uint32_t rows, uint32_t columns);

short scale_v(Vector* v, data_type scalar);
short add_mm(Matrix* M1, Matrix* M2, Matrix* sum);
short add_vv(Vector* v1, Vector* v2, Vector* sum);
short sub_vv(Vector* v1, Vector* v2, Vector* sum);
short add_vv2(Vector* v1, Vector* v2, Vector* sum);
short add_mm_new(Matrix* M1, Matrix* M2, Matrix* sum);
short add_vv_new(Vector* v1, Vector* v2, Vector* sum);
short sub_vv_new(Vector* v1, Vector* v2, Vector* sum);
short add_vv2(Vector* v1, Vector* v2, Vector* sum);

short multiply_mm(Matrix* M1, Matrix* M2, Matrix* dst);
short multiply_mv(Matrix* M, Vector* v, Vector* dst);
short multiply_mv2(Matrix* M, Vector* v, Vector* dst);
short multiply_mm_new(Matrix* M1, Matrix* M2, Matrix* dst);
short multiply_mv_new(Matrix* M, Vector* v, Vector* dst);
short multiply_mv2(Matrix* M, Vector* v, Vector* dst);

data_type vector_sqrd_mod(Vector* vector);
data_type vector_mod(Vector* vector);
void vector_print(Vector vector);

#endif
