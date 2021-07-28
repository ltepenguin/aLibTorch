#include <aLibTorch.h>
#include <memory>

class MyApp : public agl::App
{
public:
    void start() override
    {

        std::string file_name = "../my_model.pt";
        bool file_exist = agl::file_check(file_name);
        
        if(file_exist)
        {
            auto module = alt::import_pytorch_module(file_name);
            
            Tensor sample = torch::ones({1, 128, 8, 30});
            Tensor output = module->forward(sample);
            std::cout << output.sizes() << std::endl;
        }

        // Tensor my_tensor = torch::ones({1, 2, 3}, torch::kFloat32);
        // torch::save(my_tensor, "fromCPP.pt");

        // std::string file_name = "fromPython.pt";
        // bool file_exist = agl::file_check(file_name);
        // if(file_exist)
        // {
        //     torch::jit::Module container = torch::jit::load("fromPython.pt");
        //     Tensor a = container.attr("a").toTensor();
        //     std::cout << a << std::endl;
        //     Tensor b = container.attr("b").toTensor();
        //     std::cout << b << std::endl;
        //     std::string c = container.attr("c").toStringRef();
        //     std::cout << c << std::endl;
        //     int d = container.attr("d").toInt();
        //     std::cout << d << std::endl;
        // }
    }
    
    void render() override
    {
        agl::Render::plane()->draw();
    }
};

int main(int argc, char* argv[])
{
    MyApp app;
    agl::AppManager::start(&app);
    return 0;
}