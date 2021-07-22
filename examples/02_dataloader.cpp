#include <aOpenGL.h>
#include <aLibTorch.h>

// Reference: https://blog.kerrycho.com/transferlearning-libtorch/

class ImageNetDataSet : public torch::data::Dataset<ImageNetDataSet>
{
private:
    /* data */
    // Should be 2 tensors
    std::vector<torch::Tensor> states, labels;
    size_t ds_size;
public:
    ImageNetDataSet(std::string map_file)
    {
        std::tie(states, labels) = read_map(map_file);
        ds_size = states.size();
    };

    // Get 함수의 리턴 값을 상속 된 클래스의 템플릿 구조에 따라 달라 질 수 있습니다.
    // ImageNetDataSet의 경우 ImageNetDataSet 구조를 입력 하고 리턴 타입에 대해서 정의 하지 않았기 때문에
    // 기본 리턴 값은 Example의 구조로 데이터가 리턴 됩니다.
    torch::data::Example<> get(size_t index) override {
        /* This should return {torch::Tensor, torch::Tensor} */
        torch::Tensor sample_img = states.at(index);
        torch::Tensor sample_label = labels.at(index);
        return { sample_img.clone(), sample_label.clone() };
    };

    //데이터 셋의 전체 사이즈를 리턴 해주어야 합니다.
    torch::optional<size_t> size() const override {
        return ds_size;
    };
};

//Example의 기본 형은 아래 처럼 튜플 형태의 자료 형으로
// Data Tensor와 Target Tensor를 리턴 합니다.
template <typename Data = Tensor, typename Target = Tensor>
struct Example {
  using DataType = Data;
  using TargetType = Target;

  Example() = default;
  Example(Data data, Target target)
      : data(std::move(data)), target(std::move(target)) {}

  Data data;
  Target target;
};


class MyApp : public agl::App
{
public:
    void start() override
    {
        std::vector<Tensor> data;
        for(int i = 0; i < 100; ++i)
        {
            Tensor data_i = torch::tensor({i}, torch::kFloat);
            data.push_back(data_i);
        }
        Tensor stacked_data = torch::stack(data);

        torch::data::BatchDataset batch_data(data);

        // Create dataset
        static auto dataset = torch::data::datasets::MNIST("../data/fmnist/")
            .map(torch::data::transforms::Normalize<>(0.5, 0.5))
            .map(torch::data::transforms::Stack<>());
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