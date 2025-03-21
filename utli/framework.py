import numpy as np
import torch
torch.set_default_dtype(torch.float64)

# dNE Evaluation Stage
def evaluation(MA_tensor, P_tensor, B_tensor, data):
    with torch.no_grad():
        tensor_W = torch.mul(MA_tensor.repeat(1,len(MA_tensor)), B_tensor)
        tensor_WP = torch.matmul(tensor_W, P_tensor) # split ratio matrix X Path matrix
        tensor_WPD = torch.matmul(data, tensor_WP) # tensor_WP X demand vector
    return tensor_WPD

# dNE Summarization Stage
# Users may register their own TE metric functions here
def summarization(objective, L_tensor, C_tensor, data):
    with torch.no_grad():
        if objective == 'MLU':
            return torch.max(L_tensor/C_tensor)
        elif objective == 'TP':
            CLoss = torch.sum(torch.max(L_tensor-C_tensor,torch.tensor(0)))
            return 1 - CLoss/torch.sum(data)
        # Congestion loss
        else:
            CLoss = torch.sum(torch.max(L_tensor-C_tensor,torch.tensor(0)))
            return CLoss/torch.sum(data)