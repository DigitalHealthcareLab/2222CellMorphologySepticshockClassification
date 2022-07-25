import torch 

# def get_device():
#     return torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")