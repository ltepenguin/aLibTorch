Libtorch를 사용하기 위해서는 올바른 버전의 CUDA, cuDNN, driver를 다운로드 받아야 합니다. \
본 문서는 CUDA 11.1을 기준을 작성되었습니다.

## CUDA
1. CUDA 버전은 Libtorch가 지원하는 버전을 확인하고 다운로드합니다. \
https://developer.nvidia.com/cuda-11.1.0-download-archive?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu&target_version=2004&target_type=debnetwork \
마지막 명령어는 다음과 같이 수정해줘야 합니다. (버전을 명시하지 않을 경우, 최신 버전으로 다운로드해서 에러가 납니다.)
```
sudo apt-get -y install cuda-11-1
```

2. CUDA를 설치를 확인합니다.
```
nvcc --version
```
CUDA가 설치되어도 에러가 명령어를 찾지 못하는 경우가 있습니다. 다음 링크를 보고 에러를 수정하세요: https://ghostweb.tistory.com/829

## Driver
그래픽카드 드라이버가 CUDA 버전을 지원하는지 확인하세요.
```
nvidia-smi
```
여기에 보이는 CUDA version은 최대 지원 version 입니다. 따라서 다운로드한 CUDA는 해당 버전보다 낮아야 합니다.

## cuDNN
Download cuDNN: https://developer.nvidia.com/cudnn \
CUDNN이 지원하는 CUDA 버전을 꼭 확인하세요. \
CUDA 11.1 버전 기준으로 다음 버전을 다운로드 받았습니다: 
- cuDNN v8.2.1 (June 7th, 2021), for CUDA 11.x
- cuDNN Library for Linux (x86_64)

## LibTorch (Linux Version)
1. Check cuda version of your computer
```
$nvcc --version
```
2. Go to the link: https://pytorch.org/ 
</br>and download libtorch compatible with cuda version downloaded in your computer (in my case, cuda version is 11.0)

  >for cuda version 11.0, type the following commands on your command prompt
```
$ wget https://download.pytorch.org/libtorch/cu110/libtorch-cxx11-abi-shared-with-deps-1.7.0%2Bcu110.zip
```
3. Unzip the downloaded zip file (**libtorch-cxx11-abi-shared-with-deps-1.7.0+cu110.zip**)
4. Copy files inside the folder **./libtorch/** into the folder **./aLibTorch/ext/libtorch**



## Build
1. Go to the folder **aLibTorch/** and open in terminal
2. `$ mkdir build`
3. `$ cd build`
4. `$ cmake ..`
5. `$ make -j`

