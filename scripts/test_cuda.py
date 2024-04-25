import torch


if __name__ == '__main__':

    print('torch version:', torch.__version__)
    print('cuda is available:', torch.cuda.is_available())

