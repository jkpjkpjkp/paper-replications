import torch

USE_GPU = torch.cuda.is_available()
DEVICE = None

def set_device(device_id):
    global DEVICE
    DEVICE = f'cuda:{device_id}' if USE_GPU else 'cpu'
    if USE_GPU:
        torch.cuda.set_device(DEVICE)
    return DEVICE
