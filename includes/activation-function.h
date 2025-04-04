#ifndef ba9d5d_ACT
#define ba9d5d_ACT

#include <math.h>

#ifndef data_type
#define data_type float
#define data_type_str #data_type
#endif

data_type ReLU(data_type input);
data_type d_ReLU(data_type input);

data_type LReLU(data_type input);
data_type d_LReLU(data_type input);

data_type sigmoid(data_type input);
data_type d_sigmoid(data_type input);

data_type d_tanh(data_type input);

#endif
