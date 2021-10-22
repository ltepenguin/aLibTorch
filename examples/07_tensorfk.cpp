#include <aLibTorch.h>
#include <memory>

class MyApp : public agl::App
{
public:
    agl::spModel model;
    agl::Motion motion;
    agl::spKinModel kmodel;

    // model
    std::vector<Tensor> pre_Rs; // [4]
    std::vector<Tensor> pre_Ts; // [3]
    
    // animation
    std::vector<Tensor> local_Ts; // [3]
    std::vector<std::vector<Tensor>> tposes;

    // simulation
    int frame = 0;
    std::vector<Mat4> trfs;
    
    std::vector<Tensor> joint_Rs;
    std::vector<Tensor> joint_Ts;

    void start() override
    {
        std::string model_path  = "../../_data/kmodel/model/kmodel.fbx";
        std::string motion_path = "../../_data/kmodel/motion/ubi_sprint1_subject2.fbx";
        
        model  = agl::FBX(model_path).model();
        motion = agl::FBX(motion_path).motion(model).at(0);
        kmodel = agl::kinmodel(model);

        // pre-transform
        for(int i = 0; i < kmodel->pre_trfs.size(); ++i)
        {
            Mat3 Ri = kmodel->pre_trfs.at(i).block<3, 3>(0, 0);
            Quat _qi(Ri);
            Tensor qi = alt::quat_to_tquat(_qi);
            pre_Rs.push_back(qi);
            
            Tensor ti = alt::vec_to_tensor((Vec3)kmodel->pre_trfs.at(i).col(3).head<3>());
            pre_Ts.push_back(ti);
        }

        // motions
        for(auto& pose : motion.poses)
        {
            // translation
            local_Ts.push_back(alt::vec_to_tensor(pose.root_position));

            // quaternions
            std::vector<Tensor> tquats;
            for(int i = 0; i < pose.local_rotations.size(); ++i)
            {
                Tensor tquat = alt::quat_to_tquat(pose.local_rotations.at(i));
                tquats.push_back(tquat);
            }
            tposes.push_back(tquats);
        }

    }
    
    void update() override
    {
        const auto& tpose = tposes.at(frame);
        int noj = tpose.size();
        joint_Ts = std::vector<Tensor>(noj, alt::vec_to_tensor(Vec3::Zero()));
        joint_Rs = std::vector<Tensor>(noj, torch::tensor({1, 0, 0, 0}, torch::kFloat));
        
        // compute world transform
        trfs.clear();
        trfs.resize(noj, Mat4::Identity());
        const auto& idxes = kmodel->fk_order;
        const auto& pidxes = kmodel->parent_idxes;
        Tensor Ti = local_Ts.at(frame);
        // for(int i = 0; i < idxes.size(); ++i)
        for(int i = 0; i < 1; ++i)
        {
            int idx = idxes.at(i);
            int pidx = pidxes.at(i);
            std::cout << idx << "," << pidx << std::endl;

            // wolrd_trf = parent_trf * (T * Rpre) * R * S
            Tensor parent_T = (pidx >= 0) ? joint_Ts.at(pidx) : Ti;
            Tensor parent_R = (pidx >= 0) ? joint_Rs.at(pidx) : torch::tensor({1, 0, 0, 0}, torch::kFloat);
            Tensor pre_T = pre_Ts.at(idx);  // pre-translation (quat)
            Tensor pre_R = pre_Rs.at(idx);  // pre-rotation (quat)
            Tensor Ri = tpose.at(idx);       // rotation (quat)
            
            // Rotation
            Tensor R = alt::tquat_mul(pre_R, Ri);
            R = alt::tquat_mul(parent_R, R);
            //joint_Rs.at(idx) = R;
            joint_Rs.at(idx) = torch::tensor({1, 0, 0, 0}, torch::kFloat);

            // Translation
            Tensor T = alt::tquat_rot(parent_R, pre_T);
            T = T.add(parent_T);
            joint_Ts.at(idx) = T;
        }
    }
    
    void render() override
    {
        agl::Render::plane()->draw();

        //for(int i = 0; i < joint_Rs.size(); ++i)
        for(int i = 0; i < 1; ++i)
        {
            Quat q = alt::tquat_to_quat(joint_Rs.at(i));
            Vec3 T = alt::tensor_to_vec(joint_Ts.at(i));
            agl::Render::axis()
                ->orientation(q.toRotationMatrix())
                ->position(T)
                ->draw();
        }

    }
    
    void late_update() override
    {
        frame++;
    }
};

int main(int argc, char* argv[])
{
    MyApp app;
    agl::AppManager::start(&app);
    return 0;
}