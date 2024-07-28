import torch


class CudaUtil:
    @staticmethod
    def enable_debugging():
        torch.autograd.set_detect_anomaly(True)

    @staticmethod
    def enable_tf32():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    @staticmethod
    def performance():
        CudaUtil.enable_tf32()
        # Enable cuDNN benchmark mode and other fast options
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.fastest = True
        torch.backends.cudnn.deterministic = False

    @staticmethod
    def set_sead():
        seed = 42
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        #torch.backends.cudnn.deterministic = True
        #torch.backends.cudnn.benchmark = False
        #torch.backends.cudnn.enabled = True
