#pragma once
#include "torchtype.h"
#include <memory>

namespace a::lt {

class PyModule
{
public:
    explicit PyModule(std::string path);
    
    PyModule(const PyModule&) = delete;
    
    PyModule& operator=(PyModule&) = delete;
    
    Tensor forward(Tensor);
    
    void to(torch::Device);

    // member variable
    std::unique_ptr<torch::jit::Module> m_module;
};
using spPyModule = std::shared_ptr<PyModule>;

/**
 * @brief 
 * 
 * Python Code for exporting model:
 * 
 *      import torch
 * 
 *      input = torch.ones(1, 128 , 8, 30) # sample input
 *      traced_model = torch.jit.trace(model, input)
 *      traced_model.save("../model/my_model.pt")
 * 
 */
spPyModule import_pytorch_module(std::string path);

/**
 * @brief 
 * 
 * Tensor must be container's attribute!
 * 
 * Python Code:
 * 
 *      import torch
 * 
 *      class Container(torch.nn.Module):
 *          def __init__(self, tensor):
 *              super().__init__()
 *              setattr(self, "tensor", tensor)
 *      
 *      def export_tensor(tensor, path : str):
 *          container = torch.jit.script(Container(tensor))
 *          container.save(path)
 * 
 */
Tensor import_pytorch_tensor(std::string path, std::string attr = "tensor");

/**
 * @brief export to pytorch
 * 
 * Python Code:
 * 
 *      import torch
 * 
 *      def import_tensor(path : str) -> torch.Tensor:
 *          data = torch.jit.load(path)
 *          data_parameters = list(data.parameters())
 *          x : torch.Tensor = data_parameters[0]
 *          return x
 * 
 */
void export_to_pytorch(Tensor tensor, std::string path);

}