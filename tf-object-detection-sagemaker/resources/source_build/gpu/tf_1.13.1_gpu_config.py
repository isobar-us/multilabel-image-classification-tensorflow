import os
import sys

default_python_bin_path = sys.executable

os.environ['PYTHON_BIN_PATH'] = default_python_bin_path
os.environ['USE_DEFAULT_PYTHON_LIB_PATH'] = '1'
os.environ['TF_NEED_IGNITE'] = '0'
os.environ['TF_ENABLE_XLA'] = '0'
os.environ['TF_NEED_OPENCL_SYCL'] = '0'
os.environ['TF_NEED_ROCM'] = '0'
os.environ['TF_NEED_CUDA'] = '1'
os.environ['TF_DOWNLOAD_CLANG'] = '0'
os.environ['TF_NEED_MPI'] = '0'
os.environ['CC_OPT_FLAGS'] = '-march=native'
os.environ['TF_SET_ANDROID_WORKSPACE'] = '0'
os.environ['TF_CUDA_CLANG'] = '0'
os.environ['TF_NEED_TENSORRT'] = '1'
os.environ['TF_CUDA_VERSION'] = '10.0'
os.environ['TF_CUDNN_VERSION'] = '7.5.0'
os.environ['TF_NCCL_VERSION'] = '2'
os.environ['TF_CUDA_COMPUTE_CAPABILITIES'] = '3.7,7.0'
os.environ['CUDA_TOOLKIT_PATH'] = '/usr/local/cuda-10.0'
os.environ['CUDNN_INSTALL_PATH'] = '/usr/local/cuda-10.0'
os.environ['NCCL_INSTALL_PATH'] = '/usr/local/cuda-10.0'
os.environ['GCC_HOST_COMPILER_PATH'] = '/usr/bin/gcc'
os.environ['TENSORRT_INSTALL_PATH'] = '/home/ec2-user/SageMaker/TensorRT-5.0.2.6/targets/x86_64-linux-gnu'

os.environ['LD_LIBRARY_PATH'] = os.environ['LD_LIBRARY_PATH'] + ':/usr/local/cuda-10.0/lib64'
