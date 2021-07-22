#pragma once
#include <aOpenGL.h>
#include <torch/torch.h>

using Tensor         = ::torch::Tensor;
using TensorList     = ::torch::TensorList;
using TensorOptions  = ::torch::TensorOptions;
using IntArrayRef    = ::torch::IntArrayRef;

using Adam           = ::torch::optim::Adam;
using AdamOptions    = ::torch::optim::AdamOptions;
using RMSprop        = ::torch::optim::RMSprop;
using RMSpropOptions = ::torch::optim::RMSpropOptions;
using Optimizer      = ::torch::optim::Optimizer;

using nnModule       = ::torch::nn::Module;
using Linear         = ::torch::nn::Linear;
using LinearOptions  = ::torch::nn::LinearOptions;
using Conv2d         = ::torch::nn::Conv2d;
using Conv2dOptions  = ::torch::nn::Conv2dOptions;
using ConvT2d        = ::torch::nn::ConvTranspose2d;
using ConvT2dOptions = ::torch::nn::ConvTranspose2dOptions;

namespace a::lt {

torch::Tensor vec_to_tensor(VecN vec);
torch::Tensor mat_to_tensor(Mat mat);

VecN tensor_to_vec(Tensor tensor);
Mat  tensor_to_mat(Tensor tensor);

// load parameters
void load_state_dict(const nnModule* src, nnModule* dst);

}

namespace alt = a::lt;