import os
import sys

default_python_bin_path = sys.executable

os.environ['PYTHON_BIN_PATH'] = default_python_bin_path
os.environ['USE_DEFAULT_PYTHON_LIB_PATH'] = '1'
os.environ['TF_NEED_IGNITE'] = '0'
os.environ['TF_ENABLE_XLA'] = '0'
os.environ['TF_NEED_OPENCL_SYCL'] = '0'
os.environ['TF_NEED_ROCM'] = '0'
os.environ['TF_NEED_CUDA'] = '0'
os.environ['TF_DOWNLOAD_CLANG'] = '0'
os.environ['TF_NEED_MPI'] = '0'
os.environ['CC_OPT_FLAGS'] = '-march=native'
os.environ['TF_SET_ANDROID_WORKSPACE'] = '0'
