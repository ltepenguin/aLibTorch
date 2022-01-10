#pragma once
#include <aOpenGL.h>
#include <torch/torch.h>
#include <torch/csrc/jit/serialization/import.h>

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
using Conv1d         = ::torch::nn::Conv1d;
using Conv1dOptions  = ::torch::nn::Conv1dOptions;
using ConvT1d        = ::torch::nn::ConvTranspose1d;
using ConvT1dOptions = ::torch::nn::ConvTranspose1dOptions;
using Conv2d         = ::torch::nn::Conv2d;
using Conv2dOptions  = ::torch::nn::Conv2dOptions;
using ConvT2d        = ::torch::nn::ConvTranspose2d;
using ConvT2dOptions = ::torch::nn::ConvTranspose2dOptions;

namespace a::lt {

Tensor vec_to_tensor(VecN vec);
Tensor mat_to_tensor(Mat mat);

VecN tensor_to_vec(Tensor tensor);
Mat  tensor_to_mat(Tensor tensor);

// load parameters
void load_state_dict(const nnModule* src, nnModule* dst);

Tensor min_val(Tensor self, const std::vector<int>& dim);
Tensor max_val(Tensor self, const std::vector<int>& dim);

}

namespace alt = a::lt;