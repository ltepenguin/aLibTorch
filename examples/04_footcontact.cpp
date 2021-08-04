#include <aOpenGL.h>
#include <aLibTorch.h>

// ----------------------------------------------------------------------------- //

const std::vector<std::string> joint_names = {
    "Hips", 
    "LeftUpLeg", "LeftLeg", "LeftFoot", "LeftToeBase",
    "RightUpLeg", "RightLeg", "RightFoot", "RightToeBase",
    "Spine", "Spine1", "Spine2", 
    "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand", 
    "RightShoulder", "RightArm", "RightForeArm", "RightHand", 
    "Neck", "Head"
};

const std::vector<std::string> input_joint_names = {
    //"Hips", 
    //"LeftHand",
    "LeftToeBase",
    //"RightHand",
    "RightToeBase"
};

// ----------------------------------------------------------------------------- //

class EncoderImpl : public nnModule
{
public:
    Conv2d conv1, conv2, conv3, conv4, conv5, conv6, conv7, conv8;
    torch::Device dev;

    // conv size
    EncoderImpl():
        // conv1(register_module("conv1", Conv2d(Conv2dOptions( 1, 32, {3, 3}).stride({3, 1}).padding({0, 1}).bias(true)))),
        // conv2(register_module("conv2", Conv2d(Conv2dOptions(32, 64, {4, 3}).stride({1, 1}).padding({0, 1}).bias(true)))),
        conv1(register_module("conv1", Conv2d(Conv2dOptions( 1, 32, {3, 3}).stride({1, 1}).padding(1).bias(true)))),
        conv2(register_module("conv2", Conv2d(Conv2dOptions(32, 64, {3, 3}).stride({1, 1}).padding(1).bias(true)))),
        conv3(register_module("conv3", Conv2d(Conv2dOptions(64, 64, {3, 3}).stride({1, 1}).padding(1).bias(true)))),
        conv4(register_module("conv4", Conv2d(Conv2dOptions(64, 64, {3, 3}).stride({1, 1}).padding(1).bias(true)))),
        conv5(register_module("conv5", Conv2d(Conv2dOptions(64, 32, {3, 3}).stride({1, 1}).padding(1).bias(true)))),
        conv6(register_module("conv6", Conv2d(Conv2dOptions(32, 32, {3, 3}).stride({1, 1}).padding(1).bias(true)))),
        conv7(register_module("conv7", Conv2d(Conv2dOptions(32,  1, {3, 3}).stride({1, 1}).padding(1).bias(true)))),
        conv8(register_module("conv8", Conv2d(Conv2dOptions( 1,  1, {3, 3}).stride({1, 1}).padding(1).bias(true)))),
        dev(torch::kCPU)
    {
    }

    Tensor forward(Tensor x)
    {
        int winodw_size = x.sizes().back();

        //std::cout << std::endl;
        //std::cout << x.sizes() << std::endl;
        
        x = torch::sin(conv1(x));       
        //std::cout << x.sizes() << std::endl;
        x = torch::upsample_nearest2d(x, {8, winodw_size});
        x = torch::sin(conv2(x));
        //std::cout << x.sizes() << std::endl;

        x = torch::sin(conv3(x));
        x = torch::upsample_nearest2d(x, {16, winodw_size});
        x = torch::sin(conv4(x));
        //std::cout << x.sizes() << std::endl;

        x = torch::sin(conv5(x));       
        x = torch::upsample_nearest2d(x, {33, winodw_size});
        x = torch::sin(conv6(x));
        //std::cout << x.sizes() << std::endl;

        x = torch::sin(conv7(x));
        x = torch::upsample_nearest2d(x, {66, winodw_size});
        x = conv8(x);
        //std::cout << x.sizes() << std::endl;

        return x;
    }
};
TORCH_MODULE(Encoder);

// ----------------------------------------------------------------------------- //

class MyApp : public agl::App
{
public:
    agl::spModel model;
    agl::Motion motion;
    
    agl::spKinModel kmodel;
    agl::spKinMotion kmotion;

    Encoder encoder{nullptr};
    Tensor X, Y;
    Tensor Xmean, Ymean;
    Tensor Xstd, Ystd;

    struct FrameFeature
    {
        std::vector<Vec3> X;
        std::vector<Vec3> Y;
    };

    Tensor do_segment(Tensor data, int window, int stride)
    {
        int total_n = data.size(0);
        int iter = 0;
        
        std::vector<Tensor> tensors;
        tensors.reserve(total_n);
        
        while((iter + window) < total_n)
        {
            tensors.push_back(data.narrow_copy(0, iter, window));
            iter += stride;
        }
        
        return torch::stack(tensors);
    }

    std::vector<Vec3> tensor_to_v3s(Tensor data)
    {
        int noj = data.size(0);
        std::vector<Vec3> v3s;
        v3s.reserve(noj);
        for(int i = 0; i < data.size(0); i+=3)
        {
            float x = data[i + 0].item<float>();
            float y = data[i + 1].item<float>();
            float z = data[i + 2].item<float>();
            v3s.push_back(Vec3(x,y,z));
        }
        return v3s;
    }
    
    /**
     * @param inputX    [1, Xdim, window]
     * @param outputY   [1, Ydim, window]
     */
    std::vector<FrameFeature> set_segment(Tensor inputX, Tensor outputY)
    {
        std::cout << "set segments" << std::endl << std::flush;
        
        Tensor x = inputX.squeeze(0).transpose(0, 1);
        Tensor y = outputY.squeeze(0).transpose(0, 1);
        x = x.mul(Xstd).add(Xmean).contiguous(); // [window, Xdim]
        y = y.mul(Ystd).add(Ymean).contiguous(); // [window, Ydim]

        std::vector<FrameFeature> features;
        int window = x.size(0);
        features.reserve(window);
        for(int i = 0; i < window; ++i)
        {
            FrameFeature feature;
            feature.X = tensor_to_v3s(x[i]);
            feature.Y = tensor_to_v3s(y[i]);
            features.push_back(feature);
        }
        return features;
    }

    void set_results(int segment_idx)
    {
        Tensor input  = X.narrow(0, segment_idx, 1);
        Tensor output = encoder->forward(input);
        results = set_segment(input[0], output[0]);
    }
    
    int result_idx = 0;
    std::vector<FrameFeature> results;

    void start() override
    {
        const char* model_path  = "../../aOpenGL/data/fbx/kmodel/model/kmodel.fbx";
        const char* motion_path = "../../aOpenGL/data/fbx/kmodel/motion/ubi_sprint1_subject2.fbx";

        //agl::FBX model_fbx(model_path);
        //model = model_fbx.model();
       
        agl::FBX motion_fbx(motion_path);
        model  = motion_fbx.model();
        motion = motion_fbx.motion(model).at(0);

        // kinematics model
        kmodel  = agl::kinmodel(model, joint_names);
        kmotion = agl::kinmotion(kmodel, {motion});
        kmotion->init_world_basisTrf_from_shoulders("RightShoulder", "LeftShoulder");

        // extract features
        {
            //int lf_idx = kmodel->jnt_name_to_idx.at(LF);
            //int rf_idx = kmodel->jnt_name_to_idx.at(RF);
            
            std::vector<int> jnt_idxes;
            for(int i = 0; i < input_joint_names.size(); ++i)
                jnt_idxes.push_back(kmodel->jnt_name_to_idx.at(input_joint_names.at(i)));

            std::vector<Tensor> foot_features;
            std::vector<Tensor> pose_features;
            foot_features.reserve(kmotion->poses.size());
            pose_features.reserve(kmotion->poses.size());
            for(const auto& pose : kmotion->poses)
            {
                Mat4 binv = pose.world_basisTrf.inverse();
#if 0                
                Mat4 lf_trf = binv * pose.world_trfs.at(lf_idx);
                Mat4 rf_trf = binv * pose.world_trfs.at(rf_idx);
                Vec3 lf_T = lf_trf.col(3).head<3>();
                Vec3 rf_T = rf_trf.col(3).head<3>();
                auto t0 = alt::vec_to_tensor(lf_T);
                auto t1 = alt::vec_to_tensor(rf_T);
                foot_features.push_back(torch::cat({t0, t1}));
#else
                std::vector<Tensor> tensors;
                tensors.reserve(jnt_idxes.size());
                for(int j = 0; j < jnt_idxes.size(); ++j)
                {
                    Mat4 jnt_trf = binv * pose.world_trfs.at(jnt_idxes.at(j));
                    Vec3 jnt_T = jnt_trf.col(3).head<3>();
                    Tensor t_j = alt::vec_to_tensor(jnt_T);
                    tensors.push_back(t_j);
                }
                foot_features.push_back(torch::cat(tensors));
#endif
                
                
                std::vector<Tensor> jnt_Ts;
                jnt_Ts.reserve(pose.world_trfs.size());
                for(const auto& trf : pose.world_trfs)
                {
                    Vec3 pos = (binv * trf.col(3)).head<3>();
                    jnt_Ts.push_back(alt::vec_to_tensor(pos));
                }
                pose_features.push_back(torch::cat(jnt_Ts));
            }
            
            // Set data
            X = torch::stack(foot_features); // [N, dim]
            Y = torch::stack(pose_features); // [N, dim]
            
            // Normalize
            Xmean = X.mean(0);
            Ymean = Y.mean(0);
            Xstd = X.std(0);
            Ystd = Y.std(0);
            X = X.sub(Xmean).div(Xstd);
            Y = Y.sub(Ymean).div(Ystd);
            std::cout << "Xmean shape: " << Xmean.sizes() << std::endl;
            std::cout << "Ymean shape: " << Ymean.sizes() << std::endl;

            // Segment
            X = do_segment(X, 120, 15); // [N, window, dim]
            Y = do_segment(Y, 120, 15); // [N, window, dim]

            // CNN Format
            X = X.permute({0, 2, 1}); // [N, dim, window]
            Y = Y.permute({0, 2, 1}); // [N, dim, window]
            X = X.unsqueeze(1).contiguous(); // [N, 1, dim, window]
            Y = Y.unsqueeze(1).contiguous(); // [N, 1, dim, window]

            std::cout << X.sizes() << std::endl;
            std::cout << Y.sizes() << std::endl;
        }

        // Train
        encoder = Encoder();
        if(agl::file_check("encoder.pt"))
        {
            torch::load(encoder, "encoder.pt");
        }
        
        {
            auto optOption = AdamOptions().lr(0.0002);
            Adam optimizer(encoder->parameters(), optOption);

            torch::Device dev = torch::kCUDA;
            int total_iter = 100;
            int batch_size = 64;
            int data_num = X.size(0);
            auto rnd_option = TensorOptions().dtype(torch::kInt64);
            encoder->to(dev);

            for(int i = 1; i <= total_iter; ++i)
            {
                auto rnd = torch::randperm(data_num, rnd_option);
                int iter = 0;
                Tensor total_loss = torch::zeros({1}, dev);
                while(iter < data_num)
                {
                    int length = std::min(batch_size, data_num - iter);

                    auto batch_indices = rnd.narrow(0, iter, length);
                    auto input = X.index_select(0, batch_indices).to(dev);
                
                    auto output = encoder->forward(input);
                    //exit(0);
                    
                    auto output_ref = Y.index_select(0, batch_indices).to(dev);
                    auto loss = torch::nn::functional::mse_loss(output, output_ref);
                    //auto loss = torch::nn::functional::l1_loss(output, output_ref);

                    optimizer.zero_grad();
                    loss.backward();
                    optimizer.step();
                    iter += length;

                    total_loss += loss;
                }
                std::cout << i << ": " << total_loss.item<float>() << std::endl;
            }
        }
        
        encoder->to(torch::kCPU);
        torch::save(encoder, "encoder.pt");
        set_results(result_idx);
    }

    int frame = 0;

    void update() override
    {

    }

    void late_update() override
    {
        frame++;
        
        if(frame >= results.size())
        {
            result_idx += 4;
            if(result_idx >= X.size(0))
                result_idx = 0;
            set_results(result_idx);
            frame = 0;
        }
    }
    
    void render() override
    {
        agl::Render::plane()->draw();
        //agl::Render::model(model)->draw();
        
        for(auto& v3 : results.at(frame).X)
        {
            agl::Render::sphere()
                ->position(v3)
                ->scale(0.1f)->color(1, 0, 0)->draw();
        }
        for(auto& v3 : results.at(frame).Y)
        {
            agl::Render::sphere()
                ->position(v3)
                ->scale(0.1f)->color(0, 1, 1)
                ->alpha(0.5f)
                ->draw();
        }
    }
};

int main(int argc, char* argv[])
{
    MyApp app;
    agl::AppManager::start(&app);
    return 0;
}