## aOpenGL
aLibTorch는 aOpenGL이 필요합니다. 두 프로젝트는 같은 폴더 안에 있어야 합니다. \
예시: \
aOpenGL path: `~/문서/aOpenGL` \
aLibTorch path: `~/문서/aLibTorch`

## Settings

Libtorch를 사용하기 위해서는 올바른 버전의 CUDA, cuDNN, driver를 다운로드 받아야 합니다. \
본 문서는 CUDA 11.1을 기준을 작성되었습니다.

### CUDA
1. CUDA 버전은 Libtorch가 지원하는 버전을 확인하고 다운로드합니다. \
https://developer.nvidia.com/cuda-11.1.0-download-archive?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu&target_version=2004&target_type=debnetwork \
마지막 명령어는 다음과 같이 수정해줘야 합니다. (버전을 명시하지 않을 경우, 최신 버전으로 다운로드해서 버전 에러가 납니다.)
```
sudo apt-get -y install cuda-11-1
```

2. CUDA 설치를 확인합니다.
```
nvcc --version
```
CUDA가 설치되어도 명령어를 찾지 못하는 경우가 있습니다. 다음 링크를 보고 에러를 수정하세요:
https://askubuntu.com/questions/885610/nvcc-version-command-says-nvcc-is-not-installed

### Driver
그래픽카드 드라이버가 CUDA 버전을 지원하는지 확인하세요.
```
nvidia-smi
```
> nvidia-smi에 보이는 CUDA version은 최대 지원 version 입니다. 따라서 driver에 표시된 버전은 다운로드한 CUDA 버전보다 높아야 합니다.

### cuDNN
Download cuDNN: https://developer.nvidia.com/cudnn \
CUDNN이 지원하는 CUDA 버전을 꼭 확인하세요. \
CUDA 11.1 버전 기준으로 다음 버전을 다운로드합니다: 
> 버전: cuDNN v8.2.1 (June 7th, 2021), for CUDA 11.x \
> cuDNN Library for Linux (x86_64) 으로 다운로드

설치는 다음 링크를 따릅니다: https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html
1. cuDNN 다운로드한 폴더를 엽니다.
2. cuDNN을 압축 해제합니다.
```
tar -xzvf cudnn-x.x-linux-x64-v8.x.x.x.tgz
```
3. CUDA tookit directory로 파일들을 복사하기위해 다음 명
```
sudo cp cuda/include/cudnn*.h /usr/local/cuda/include 
sudo cp -P cuda/lib64/libcudnn* /usr/local/cuda/lib64 
sudo chmod a+r /usr/local/cuda/include/cudnn*.h /usr/local/cuda/lib64/libcudnn*
```

### LibTorch
1. LibTorch를 다운로드합니다: https://pytorch.org/
2. 압축 해제합니다.
3. 압축 해제한 `libtorch` 폴더 안의 파일들을 `aLibTorch/ext/libtorch` 폴더로 옮깁니다.

## Build
다음 명령어들을 순차적으로 실행합니다.
```
mkdir build
cd build
cmake ..
make -j
```


