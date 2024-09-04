import sys
import subprocess

def check_package(package_name):
    try:
        __import__(package_name)
        return True
    except ImportError:
        return False

def get_package_version(package_name):
    try:
        return __import__(package_name).__version__
    except:
        return "Unknown"

def check_cuda():
    try:
        output = subprocess.check_output(['nvidia-smi']).decode('utf-8')
        return "NVIDIA-SMI" in output
    except:
        return False

def check_unified_memory(device):
    try:
        return device.get_attributes().get(cuda.device_attribute.UNIFIED_ADDRESSING) == 1
    except:
        return False

def main():
    print("Checking for required GPU packages:")
    
    # Check for CuPy
    cupy_available = check_package("cupy")
    print(f"CuPy: {'Installed' if cupy_available else 'Not installed'}")
    if cupy_available:
        print(f"  Version: {get_package_version('cupy')}")
        import cupy as cp
        try:
            print(f"  CUDA version: {cp.cuda.runtime.runtimeGetVersion()}")
            print(f"  Number of GPUs: {cp.cuda.runtime.getDeviceCount()}")
            for i in range(cp.cuda.runtime.getDeviceCount()):
                props = cp.cuda.runtime.getDeviceProperties(i)
                print(f"  GPU {i}: {props['name'].decode()}")
                print(f"    Unified Memory: {'Supported' if props['managedMemory'] else 'Not supported'}")
        except:
            print("  Unable to retrieve GPU information")
    
    # Check for PyCUDA
    pycuda_available = check_package("pycuda")
    print(f"PyCUDA: {'Installed' if pycuda_available else 'Not installed'}")
    if pycuda_available:
        print(f"  Version: {get_package_version('pycuda')}")
        import pycuda.driver as cuda
        cuda.init()
        print(f"  Number of GPUs: {cuda.Device.count()}")
        for i in range(cuda.Device.count()):
            dev = cuda.Device(i)
            print(f"  GPU {i}: {dev.name()}")
            print(f"    Unified Memory: {'Supported' if check_unified_memory(dev) else 'Not supported'}")
    
    # Check CUDA availability
    cuda_available = check_cuda()
    print(f"CUDA: {'Available' if cuda_available else 'Not available'}")
    
    # Summary
    if cupy_available and pycuda_available and cuda_available:
        print("\nAll required packages are installed and CUDA is available.")
        print("Your system should be ready for GPU-accelerated calculations.")
    else:
        print("\nSome required components are missing:")
        if not cupy_available:
            print("- CuPy needs to be installed")
        if not pycuda_available:
            print("- PyCUDA needs to be installed")
        if not cuda_available:
            print("- CUDA is not available or not properly configured")
        print("Please install missing packages and ensure CUDA is properly set up for GPU acceleration.")

if __name__ == "__main__":
    main()