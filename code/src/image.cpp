#include "aLibTorch/image.h"

namespace a::lt {

unsigned char* _tensor_to_data(Tensor tensor)
{
    // C x H x W -> H x W x C
    tensor = tensor.permute({1, 2, 0});
    tensor = tensor.to(torch::kCPU).flatten().to(torch::kUInt8).contiguous();
    int N  = tensor.size(0);
    unsigned char* tensor_data = tensor.data_ptr<unsigned char>();
    unsigned char* data = (unsigned char*)std::calloc(N, sizeof(unsigned char));
    std::memcpy(data, tensor_data, sizeof(unsigned char) * N);
    return data;
}

Tensor image_scale(Tensor tensor, int w_scale, int h_scale)
{
    if(w_scale > 1)
    {
        tensor = tensor.unsqueeze(0);                   // 1 x C x H x W
        tensor = tensor.repeat_interleave(w_scale, 0);  // s x C x H x W
        tensor = tensor.permute({1, 2, 3, 0});          // C x H x W x s
        tensor = tensor.flatten(2, 3);                  // C x H x Ws
    }
    if(h_scale > 1)
    {
        tensor = tensor.unsqueeze(0);                   // 1 x C x H x W
        tensor = tensor.repeat_interleave(h_scale, 0);  // s x C x H x W
        tensor = tensor.permute({1, 2, 0, 3});          // C x H x s x W
        tensor = tensor.flatten(1, 2);                  // C x Hs x W
    }

    return tensor.contiguous();
}

Tensor image_resize(Tensor tensor, int width, int height)
{
    assert(tensor.dim() == 3);
    bool same_h = (tensor.size(1) == height);
    bool same_w = (tensor.size(2) == width);
    if(same_h && same_w)
        return tensor;

    auto dtype = tensor.dtype();
#if 0
    // upsampling only
    tensor = tensor.unsqueeze(0).to(torch::kFloat);
    tensor = torch::upsample_nearest2d(tensor, {height, width});
    tensor = tensor.squeeze(0).to(dtype);
#else
    namespace F = torch::nn::functional;
    tensor = tensor.unsqueeze(0).to(torch::kFloat);
    auto option = F::InterpolateFuncOptions()
        .size(std::vector<int64_t>({height, width}))
        //.mode(torch::kBilinear)
        //.align_corners(false);
        .mode(torch::kNearest);
    tensor = F::interpolate(tensor, option);
    tensor = tensor.squeeze(0).to(dtype);
#endif
    return tensor.contiguous();
}

agl::Image::spData create_image(Tensor tensor, std::string name)
{
    int C = tensor.size(0);
    int H = tensor.size(1);
    int W = tensor.size(2);

    tensor = tensor.contiguous();
    unsigned char* data = _tensor_to_data(tensor);
    auto image = agl::Image::create(name, data, W, H, C);
    return image;
}

agl::Texture create_texture(Tensor tensor, std::string name, int width, int height, bool gl_nearest)
{
    if(tensor.dim() != 3)
    {
        std::cerr << name << " texture error! Tensor dim must be 3: " << tensor.sizes() << std::endl;
        assert(tensor.dim() == 3);
    }
    
    if(tensor.size(0) == 1)
    {
        tensor = tensor.repeat_interleave(3, 0);
    }

    tensor = tensor.contiguous();
    tensor = image_resize(tensor, width, height).set_requires_grad(false);
    auto image = create_image(tensor, name);
    return agl::TextureLoader::create(name, image, gl_nearest);
}

}