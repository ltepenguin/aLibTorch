#include "aLibTorch/textfile.h"
#include <fstream>

namespace a::lt {

void save_tensor_as_text(Tensor tensor, std::string file_name)
{
    tensor = tensor.to(torch::kFloat).contiguous();
    const float* ptr = (float*)tensor.data_ptr();

    assert(tensor.dim() < 3);

    int dim = tensor.dim();
    int line_length = tensor.sizes().back();
    int total_n = (dim == 2) ? tensor.size(0) * tensor.size(1) : tensor.size(0);
    
    std::ofstream writeFile(file_name.c_str());
    if(writeFile.is_open())
    {
        int iter = 0;
        while(iter < total_n)
        {
            writeFile << ptr[iter] << "\t";
            iter++;
            if(iter % line_length == 0)
                writeFile << "\n";
        }
        writeFile.close();
    }

}
}