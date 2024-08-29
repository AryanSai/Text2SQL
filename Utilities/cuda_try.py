import torch

print("sairam")

def check_cuda():
    if torch.cuda.is_available():
        print("CUDA is available. Using GPU.")
        print(f"Device name: {torch.cuda.get_device_name(0)}")
        print(f"Memory allocated: {torch.cuda.memory_allocated(0) / 1024 ** 2:.2f} MB")
        print(f"Memory cached: {torch.cuda.torch.cuda.memory_reserved(0) / 1024 ** 2:.2f} MB")
    else:
        print("CUDA is not available. Using CPU.")

check_cuda()
