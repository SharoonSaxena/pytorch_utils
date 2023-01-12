import torch


def cpu():
    """Returns CPU as target device"""
    return torch.cuda.device("cpu")


def gpu(i=0):
    """Returns i-th gpu"""
    return torch.cuda.device(f"cuda:{i}")


def num_gpus():
    """Returns number of available GPUs"""
    return torch.cuda.device_count()


def try_gpu(i=0):
    """Return gpu(i) if exists, otherwise return cpu()."""
    if num_gpus() >= i + 1:
        return gpu(i)
    return cpu()
