#include <aOpenGL.h>
#include <aLibTorch.h>
#include <iostream>
#include <cmath>

// Configuration ----------------------------------------------------------------------- //
const char* model_path  = "../data/fbx/kmodel/model/kmodel.fbx";
const char* motion_path = "../data/fbx/kmodel/motion/ubi_sprint1_subject2.fbx";

std::vector<std::string> important_joint_names = 
{
    "Hips", 
    "LeftUpLeg", "LeftLeg", "LeftFoot", "LeftToeBase",
    "RightUpLeg", "RightLeg", "RightFoot", "RightToeBase",
    "Spine", "Spine1", "Spine2", 
    "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand", 
    "RightShoulder", "RightArm", "RightForeArm", "RightHand", 
    "Neck", "Head"
};
// ------------------------------------------------------------------------------------- //

class EncoderImpl : public nnModule
{
public:
    Linear linear1, linear2;

    EncoderImpl(int input_dim, int hidden_dim, int output_dim):
        linear1(register_module("linear1", Linear(LinearOptions(input_dim, hidden_dim).bias(true)))),
        linear2(register_module("linear2", Linear(LinearOptions(hidden_dim, output_dim).bias(true))))
    {
    };

    Tensor forward(Tensor x)
    {
        x = linear1(x);
        x = torch::relu(x);
        x = torch::relu(linear2(x));
        return x;
    }
};
TORCH_MODULE(Encoder);

class DecoderImpl : public nnModule
{
public:
    Linear linear1, linear2;
    Tensor hidden_state;

    DecoderImpl(int input_dim, int hidden_dim, int output_dim):
        linear1(register_module("linear1", Linear(LinearOptions(input_dim, hidden_dim).bias(true)))),
        linear2(register_module("linear2", Linear(LinearOptions(hidden_dim, output_dim).bias(true)))),
        hidden_state(register_parameter("hidden_tensor", torch::tensor({0.0f, 1.0f})))
    {
    };

    Tensor forward(Tensor x)
    {
        x = torch::relu(linear1(x));
        x = linear2(x);
        return x;
    }
};
TORCH_MODULE(Decoder);

// ------------------------------------------------------------------------------------- //
class Test : public agl::App
{
public:
    Encoder encoder{nullptr};
    Decoder decoder{nullptr};
    Adam*   optimizer{nullptr};

    agl::spModel gl_model;

    void start() override
    {
        agl::FBX model_fbx(model_path);
        agl::FBX motion_fbx(motion_path);
        gl_model = motion_fbx.model();

        // Dataset
        int data_n = 1000;
        Tensor data = torch::rand({data_n, 66});

        // Network model
        encoder = Encoder(66, 128, 128);
        decoder = Decoder(128, 128, 66);

        // Load network?
        bool file_exist = agl::file_check("my_encoder.pt");
        if(file_exist)
        {
            torch::load(encoder, "my_encoder.pt");
        }

        // Set optimizer
        auto optOption = AdamOptions().lr(1e-4);
        optimizer = new Adam(encoder->parameters(), optOption);
        optimizer->add_param_group(decoder->parameters());

        int batch_size = 64;
        
        TensorOptions tensor_option = TensorOptions().dtype(torch::kInt32).device(torch::kCPU);

        for(int epoch = 0; epoch < 100; ++epoch)
        {
            Tensor random_indices = torch::randperm(data_n, tensor_option); // 0 ~ data_num 사이를 랜덤하게 shuffle
            int iter = 0;
            
            Tensor epoch_loss_sum = torch::zeros({1}, torch::kCPU);

            while(iter < data_n)
            {
                // 마지막 batch도 쓸 경우
                int    iter_batch_size = std::min(batch_size, data_n - iter);
                Tensor iter_batch_indices = random_indices.narrow(0, iter, iter_batch_size);    // [batch size]. integer
                Tensor mini_batch_data = data.index_select(0, iter_batch_indices);              // [batch size, feature dim]. float
                
                // Forward
                Tensor x = torch::rand({22 * 3}).to(torch::kCPU);
                Tensor z = encoder->forward(x);
                Tensor xhat = decoder->forward(z);

                // Loss
                Tensor loss = torch::nn::functional::mse_loss(xhat, x);
                epoch_loss_sum.add_(loss);

                // Backpropagation
                optimizer->zero_grad();
                loss.backward();
                optimizer->step();        

                // Next iterate
                iter += batch_size;
            }

            // Print loss
            std::cout << "epoch : " << epoch << ", loss sum: " << epoch_loss_sum << std::endl;

            // Save model
            torch::save(encoder, "my_encoder.pt");
            torch::save(decoder, "my_decoder.pt");
        }
    }

    void update() override
    {
    }

    void render() override
    {
    }


    void key_callback(char key, int action) override
    {
        if(action != GLFW_PRESS)
            return;
        
        if(key == '1')
            this->capture(true);
        if(key == '2')
            this->capture(false);
    }
    
};

int main(int argc, char* argv[])
{
    Test app;
    agl::AppManager::start(&app);
    return 0;
} 