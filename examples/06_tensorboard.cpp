#include <aLibTorch.h>
#include <memory>
#include <tensorboard_logger.h>

class MyApp : public agl::App
{
public:
    void start() override
    {
        TensorBoardLogger logger("tb/tfevents.pb");
        for(int i = 0; i < 10; ++i)
            logger.add_scalar("test0", i, i * 1.0f);
        for(int i = 0; i < 10; ++i)
            logger.add_scalar("test1", i, i * 2.0f);

        TensorBoardLogger logger2("tb2/tfevents.pb");
        for(int i = 0; i < 10; ++i)
            logger2.add_scalar("test0", i, i * 1.5f);
        for(int i = 0; i < 10; ++i)
            logger2.add_scalar("test1", i, i * 2.5f);

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