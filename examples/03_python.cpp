#include <aLibTorch.h>
#include <memory>

class MyApp : public agl::App
{
public:
    void start() override
    {
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