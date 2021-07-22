#include <aOpenGL.h>
#include <aLibTorch.h>

#define DEV torch::kCUDA

// -------------------------------------------------------------------------- //

class EncoderImpl : public nnModule
{
public:
    Linear linear1, linear2, linear3;
    torch::Device dev;

    EncoderImpl():
        linear1(register_module("linear1", Linear(LinearOptions(784, 128).bias(true)))),
        linear2(register_module("linear2", Linear(LinearOptions(128, 64).bias(true)))),
        linear3(register_module("linear3", Linear(LinearOptions( 64, 32).bias(true)))),
        dev(torch::kCPU)
    {
    }

    Tensor forward(Tensor x)
    {
        x = torch::relu(linear1(x));
        x = torch::relu(linear2(x));
        x = torch::tanh(linear3(x));
        return x;
    }
};
TORCH_MODULE(Encoder);

// -------------------------------------------------------------------------- //

class DecoderImpl : public nnModule
{
public:
    Linear linear1, linear2, linear3;
    torch::Device dev;

    DecoderImpl():
        linear1(register_module("linear1", Linear(LinearOptions( 32,  64).bias(true)))),
        linear2(register_module("linear2", Linear(LinearOptions( 64, 128).bias(true)))),
        linear3(register_module("linear3", Linear(LinearOptions(128, 784).bias(true)))),
        dev(torch::kCPU)
    {
    }

    Tensor forward(Tensor x)
    {
        x = torch::relu(linear1(x));
        x = torch::relu(linear2(x));
        x = torch::tanh(linear3(x));
        return x;
    }
};
TORCH_MODULE(Decoder);

// -------------------------------------------------------------------------- //

class MyApp : public agl::App
{
public:
    
    Encoder encoder{nullptr};
    Decoder decoder{nullptr};
    Adam*   optimizer{nullptr};
    
    void start() override
    {
        torch::manual_seed(2021);

        // Initialize training settings
        encoder = Encoder();
        decoder = Decoder();

        // set optimizer
        auto optOption = AdamOptions().lr(1e-4);
        optimizer = new Adam(encoder->parameters(), optOption);
        optimizer->add_param_group(decoder->parameters());
        
        // initialize
        encoder->to(DEV);
        decoder->to(DEV);
        iterate(0, false);
    }

    void iterate(int interate_n, bool change_texture)
    {
        // dataset
        static auto dataset = 
            torch::data::datasets::MNIST("../data/mnist/")
            .map(torch::data::transforms::Normalize<>(0.5, 0.5))
            .map(torch::data::transforms::Stack<>());
        
        static int64_t kBatchSize = 32;

        static auto data_loader = torch::data::make_data_loader(
            std::move(dataset),
            torch::data::DataLoaderOptions().batch_size(kBatchSize).workers(2));
        
        static auto iterator = data_loader->begin();
        static int epoch = 0;

        static Tensor test_batch_imgs;

        static const std::vector<int64_t> data_size = {kBatchSize, 1, 28, 28};
        static const std::vector<int64_t> fc_size   = {kBatchSize, 1, 784};
        
        for(int i = 0; i < interate_n; ++i)
        {
            // get batch
            Tensor batch_imgs = iterator->data;              // batch x 1 x 28 x 28
            Tensor input = batch_imgs.to(DEV).flatten(2, 3); // batch x 1 x 784
            
            // loss
            Tensor latent_z = encoder->forward(input);
            Tensor output = decoder->forward(latent_z);
            Tensor loss = torch::nn::functional::mse_loss(input, output);
            
            // backpropagation
            optimizer->zero_grad();
            loss.backward();
            optimizer->step();

            // next iterator
            ++iterator;
            if(iterator == data_loader->end())
            {
                iterator = data_loader->begin();
                ++epoch;
                std::cout << "epoch: " << epoch << std::endl << std::flush;
            }

            // visualize
            if(i == 0)
            {
                if(change_texture)
                {
                    test_batch_imgs  = batch_imgs.to(torch::kCPU);
                }
                
                Tensor test_input = test_batch_imgs.to(DEV).flatten(2, 3);
                Tensor test_latent_z = encoder->forward(test_input);
                Tensor test_output   = decoder->forward(test_latent_z);
                Tensor test_output_imgs = test_output.to(torch::kCPU).reshape(data_size);

                // value range: 0 ~ 255.0f
                Tensor input_rgb_imgs  = 255.0f * 0.5f * (test_batch_imgs + 1.0f);
                Tensor output_rgb_imgs = 255.0f * 0.5f * (test_output_imgs + 1.0f);
                Tensor latent_rgb_imgs = 255.0f * 0.5f * (test_latent_z.unsqueeze(1) + 1.0f);

                alt::create_texture( input_rgb_imgs[0],  "input", 256, 256, true);
                alt::create_texture(latent_rgb_imgs[0], "latent", 256, 256, true);
                alt::create_texture(output_rgb_imgs[0], "output", 256, 256, true);
            }
        }
    }

    int frame = 0;
    void update() override
    {
        camera().set_perspective(false);
        camera().set_position(Vec3(0, 0, 1));
        camera().set_focus(Vec3(0, 0, 0));

        if(frame == 0)
            iterate(10, true);
        else
            iterate(10, false);
        
        frame = (frame + 1) % 300;
    }

    void render() override
    {
        static const Mat3 R0(AAxis(M_PI * 0.5f, Vec3::UnitZ()));
        static const Mat3 R1(AAxis(M_PI * 0.5f, Vec3::UnitX()));

        agl::Render::plane()
            ->position(-0.8f, 0, 0)
            ->orientation(R1)
            ->texture("input")
            ->debug(true)
            ->draw();

        agl::Render::plane()
            ->position(-0.0f, 0, 0)
            ->orientation(R0 * R1)
            ->texture("latent")
            ->scale(1.0f, 1.0f, 0.1f)
            ->debug(true)
            ->draw();
        
        agl::Render::plane()
            ->position(0.8f, 0, 0)
            ->orientation(R1)
            ->texture("output")
            ->debug(true)
            ->draw();
    }
};

int main(int argc, char* argv[])
{
    MyApp app;
    agl::AppManager::start(&app);
    return 0;
}