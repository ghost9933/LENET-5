import torch
print("PyTorch Version:", torch.__version__)
print("CUDA Available:", torch.cuda.is_available())
print("CUDA Version:", torch.version.cuda)
print("cuDNN Version:", torch.backends.cudnn.version())

def check_cuda():
    # Check if CUDA is available
    if torch.cuda.is_available():
        print("CUDA is available. PyTorch can use the GPU!")
        
        # Print GPU details
        print(f"Device Name: {torch.cuda.get_device_name(0)}")
        print(f"Device Count: {torch.cuda.device_count()}")
        print(f"Current Device: {torch.cuda.current_device()}")
        
        # Create a tensor and move it to the GPU
        tensor = torch.rand(3, 3).to("cuda")
        print(f"Tensor on device: {tensor.device}")

        # Run a simple operation on the GPU
        tensor = tensor * 2
        print("Sample operation result on GPU tensor:")
        print(tensor)
    else:
        print("CUDA is not available. Running on CPU.")
        
# Run the check
check_cuda()
