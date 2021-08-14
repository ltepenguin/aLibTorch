#include "aLibTorch/torchtype.h"

namespace a::lt {

Tensor vec_to_tensor(VecN vec)
{
    auto tensor = torch::zeros({vec.rows()});
    float* data = tensor.data_ptr<float>();

    Eigen::Map<VecN> eigen_map(data, tensor.size(0), 1);
    eigen_map = vec.cast<float>();
    tensor.requires_grad_(false);
    
    return tensor;
}

Tensor mat_to_tensor(Mat mat)
{
    auto tensor = torch::zeros({mat.cols(),mat.rows()});
    float* data = tensor.data_ptr<float>();

    Eigen::Map<Mat> eigen_map(data, tensor.size(1), tensor.size(0));
    eigen_map = mat.cast<float>();
    tensor.requires_grad_(false);

    return tensor.transpose(0,1);
}

VecN tensor_to_vec(Tensor tensor)
{
    assert(tensor.dim() == 1);
    tensor = tensor.contiguous().toType(torch::kFloat);
    
    int n = tensor.size(0);
    VecN vec(n);
    std::memcpy(vec.data(), tensor.data_ptr(), sizeof(float)* n);
    return vec;
}

Mat tensor_to_mat(Tensor tensor)
{
    assert(tensor.dim() == 2);
    tensor = tensor.contiguous().toType(torch::kFloat);
    
    int row = tensor.size(0);
    int col = tensor.size(1);
    
    Mat mat(col, row);
    std::memcpy(mat.data(), tensor.data_ptr(), sizeof(float)* row * col);
    return mat.transpose();
}

void load_state_dict(const nnModule* src, nnModule* dst)
{
    torch::NoGradGuard guard;
    
    using tDict = torch::OrderedDict<std::string, Tensor>;
    tDict params  = dst->named_parameters(true);
    tDict buffers = dst->named_buffers(true);
    const tDict newParams  = src->named_parameters(true);
    const tDict newBuffers = src->named_buffers(true);

    for(auto& val: newParams)
    {
        auto name = val.key();
        auto* t = params.find(name);
        if(t != nullptr)
        {
            t->copy_(val.value());
        }
    }
    
    for(auto& val: newBuffers)
    {
        auto name = val.key();
        auto* t = buffers.find(name);
        if( t!= nullptr)
        {
            t->copy_(val.value());
        }
    }
    
    return;
}

Tensor min_val(Tensor self, const std::vector<int>& dim)
{
    Tensor val = self;
    Tensor indices;
    for(int i = 0; i < dim.size(); ++i)
    {
        std::tie(val, indices) = val.min(dim.at(i), true);
    }
    return val.squeeze();
}

Tensor max_val(Tensor self, const std::vector<int>& dim)
{
    Tensor val = self;
    Tensor indices;
    for(int i = 0; i < dim.size(); ++i)
    {
        std::tie(val, indices) = val.max(dim.at(i), true);
    }
    return val.squeeze();
}

}