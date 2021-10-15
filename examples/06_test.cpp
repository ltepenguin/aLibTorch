#include <aLibTorch.h>
#include <memory>
#include <tensorboard_logger.h>

class MyApp : public agl::App
{
public:
    int test_log_scalar(TensorBoardLogger& logger) 
    {
        std::cout << "test log scalar" << std::endl;
        for (int i = 0; i < 1000; ++i) 
        {
            logger.add_scalar("scalar", i, i * 1.0f);
        }
        return 0;
    }


    void start() override
    {
        TensorBoardLogger logger("tb/tfevents.pb");
        test_log_scalar(logger);
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