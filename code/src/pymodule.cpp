#include <aLibTorch/pymodule.h>

namespace a::lt {

PyModule::PyModule(std::string path)
{
    m_module = std::make_unique<torch::jit::Module>(torch::jit::load(path));
}

Tensor PyModule::forward(Tensor x)
{
    std::vector<torch::jit::IValue> input = {x};
    return m_module->forward(input).toTensor();
}

void PyModule::to(torch::Device dev)
{
    m_module->to(dev);
}

spPyModule import_pytorch_module(std::string path)
{
    return std::make_shared<PyModule>(path);
}

Tensor import_pytorch_tensor(std::string path, std::string attr)
{
    auto module = torch::jit::load(path);
    return module.attr(attr).toTensor();
}

void export_to_pytorch(Tensor tensor, std::string path)
{
    torch::save(tensor, path);
}

}