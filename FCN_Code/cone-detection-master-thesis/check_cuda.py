import torch

if __name__ == "__main__":
    if torch.cuda.is_available():
        print("CUDA is availiable")
    else:
        print("CUDA is NOT availiable.")
        print("Calculation will be running on the CPU.")
        print("Please, check that CUDA SDK is installed.") 
        print("Please, check that 'torch' 'torchvision' 'torchaudio' is correctly installed with corresponding CUDA version.")