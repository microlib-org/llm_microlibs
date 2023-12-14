import torch


def get_cpu_memory():
    with open('/proc/meminfo', 'r') as meminfo:
        for line in meminfo:
            if 'MemAvailable' in line:
                available_memory_kb = int(line.split()[1])
                # Convert from KB to GB
                available_memory_gb = available_memory_kb / (1024 ** 2)
                return available_memory_gb
    return 0


def get_gpu_memory(device):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. No GPU detected.")

    gpu_index = int(device.split(':')[1])
    if gpu_index >= torch.cuda.device_count():
        raise ValueError("Invalid CUDA device index.")
    return torch.cuda.get_device_properties(gpu_index).total_memory / (1024 ** 3)


def is_cpu_memory_less_than_gpu(device):
    available_cpu_memory_gb = get_cpu_memory()
    free_gpu_memory_gb = get_gpu_memory(device)
    return available_cpu_memory_gb < free_gpu_memory_gb
