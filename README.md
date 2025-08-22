# workstation-benchmarking
Python package to benchmark gpu output and thermals, designed for the Thought family of workstations in the Bionic Vision Lab, but will likely work elsewhere.

Included benchmarks: 
- [pytorch-benchmark](https://github.com/pytorch/benchmark): Pytorch managed test suite for GPU and CPU training and eval of common models, on smaller toy datasets.
- [torchvision imagenet reference](https://github.com/pytorch/vision/tree/main/references/classification): Torchvision reference script for training on ImageNet, to test full scale training and disk IO. Must have ImageNet on disk at $IMAGENET.
- [stable-diffusion](https://github.com/yachty66/gpu-benchmark): Uses gpu-benchmark to test number of images generated in 5 minutes
- [gpu-burn](https://github.com/wilicc/gpu-burn): Uses max throughput CUDA kernels to push GPU thermals and test raw GFlop/s

## Usage
`python3 run_benchmarks.py -v`

## Installation
It is recommended to use a standalone conda environment. Python 3.11 and Cuda 12.3 are tested.
You must install each of the benchmark platforms. These four were chosen to be as easy to install as possible. 
You may opt to skip any given benchmark, and pass e.g. `--skip-gpu-burn` to run_benchmarks.py. 

There is a bundled script, install.sh, which attempts to automatically install them all. This works on some platforms at the time of 
publishing this repo, but it likely will not work on every platform or long in the future.
`bash install.sh`
Instead, this script is meant to serve as a starting point for installing each of the required packages, in combination with official installation instructions 
for each platform.  


## Results
Results are automatically uploaded to this google sheet (recommended), and can also be output to CSV with the `--out-csv` flag. 
