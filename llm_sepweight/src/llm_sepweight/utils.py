def get_available_memory_linux():
    with open('/proc/meminfo', 'r') as meminfo:
        for line in meminfo:
            if 'MemAvailable' in line:
                available_memory_kb = int(line.split()[1])
                # Convert from KB to bytes
                available_memory = available_memory_kb * 1024
                return available_memory
    return 0

def is_cpu_memory_less_than_gpu(device: str):
    # Check if CUDA is available and the specified device is valid
    if not torch.cuda.is_available() or not torch.cuda.device_count() > int(device.split(':')[1]):
        raise ValueError("Invalid CUDA device or CUDA not available.")

    # Get GPU memory information
    gpu_index = int(device.split(':')[1])
    total_gpu_memory = torch.cuda.get_device_properties(gpu_index).total_memory
    free_gpu_memory = total_gpu_memory - torch.cuda.memory_allocated(gpu_index)

    # Get CPU memory information using Linux-specific method
    available_cpu_memory = get_available_memory_linux()

    # Compare and return
    return available_cpu_memory < free_gpu_memory
