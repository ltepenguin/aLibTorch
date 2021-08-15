## CUDA
1. Download cuda: https://developer.nvidia.com/cuda-11.0-download-archive?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu&target_version=2004&target_type=debnetwork.
The last command should be:
```
sudo apt-get install cuda-11-0
```
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

