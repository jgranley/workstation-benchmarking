echo "Install script for benchmarking suite"
echo "Note, this is not meant to be a one-click solution script.
At the time of writing, this installs everything properly. But things change.
Instead, it should be used as a reference. If it runs, great, otherwise, read 
through each section and use it as a starting point, in combination with official 
tool-specific documentation, in order to install each benchmark"

echo
echo "Current supported benchmarks: [gpu-burn, gpu-benchmark, pytorch-benchmark, torchvision-data-io]"

TORCH_CUDA_VERSION="126"
LOCAL_CUDA_VERSION=$(nvcc --version | grep -oP 'V\K[0-9]+\.[0-9]+' | head -n 1)

echo "Please confirm the following settings:
- Conda environment: $(conda info --envs | grep '*' | awk '{print $1}')
- Working directory: $(pwd)
- Torch CUDA version (will be installed if not already): $TORCH_CUDA_VERSION
- Local CUDA version: $LOCAL_CUDA_VERSION
"
read -p "Press any key to continue or Ctrl+C to cancel..." -n1 -s
echo


### gpu-burn ###
echo -n "Install gpu-burn? ([Y]/n): "
read -r install_gpu_burn
install_gpu_burn=${install_gpu_burn:-y}
if [[ $install_gpu_burn == [yY] ]]; then
    echo "Installing gpu-burn..."
    git clone https://github.com/wilicc/gpu-burn
    cd gpu-burn
    # docker build -t gpu_burn .
    compute=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1)
    make clean
    make COMPUTE=$compute
    cd ..
else
    echo "Skipping gpu-burn installation."
fi



### gpu-benchmark ###
# check if repo has changed if this install fails
echo "Install gpu-benchmark? ([Y]/n): "
read -r install_gpu_benchmark
install_gpu_benchmark=${install_gpu_benchmark:-y}
if [[ $install_gpu_benchmark == [yY] ]]; then
    pip install git+https://github.com/jgranley/gpu-benchmark
else
    echo "Skipping gpu-benchmark installation."
fi


### pytorch-benchmark ###
echo "Install pytorch-benchmark? ([Y]/n): "
read -r install_pytorch_benchmark
install_pytorch_benchmark=${install_pytorch_benchmark:-y}
if [[ $install_pytorch_benchmark == [yY] ]]; then
    # check if they have numba already
    has_numba=$(pip list | grep numba | wc -l)
    if [[ $has_numba -eq 0 ]]; then
        echo "Numba is not installed. Installing..."
        pip install numba
    fi
    conda install -y -c pytorch magma-cuda121
    pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu$TORCH_CUDA_VERSION
    git clone https://github.com/pytorch/benchmark
    cd benchmark
    python3 install.py
    cd ..
else
    echo "Skipping pytorch-benchmark installation."
fi


### torchvision data IO testing ###
echo "Install torchvision data IO testing? ([Y]/n): "
read -r install_torchvision_data_io
install_torchvision_data_io=${install_torchvision_data_io:-y}
if [[ $install_torchvision_data_io == [yY] ]]; then
    hastorch=$(pip list | grep torchvision | wc -l)
    if [[ $hastorch -eq 0 ]]; then
        echo "Torchvision is not installed. Installing..."
        pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu$TORCH_CUDA_VERSION
    fi
    git clone https://github.com/pytorch/vision
    # verify imagenet
    echo "Assuming imagenet is located at $IMAGENET. If it is not, please download and set the IMAGENET environment variable."
else
    echo "Skipping torchvision data IO testing installation."
fi

echo "Installation complete."