# version 280325 - transformer 4.3.6

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from torch.nn.functional import mse_loss

import numpy as np

torch.set_default_dtype(torch.float32)
global_dtype = torch.float32
# global_np_dtype = np.float32

class MLModel():
    def __init__(self, PATH, model_type = "Transformer_v43"):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model_type = model_type
        self.field_base = 8

        self.criterion = nn.MSELoss()

        if model_type == "Transformer_v43":
            # Transformer parameters are defined in transformer file
            from transformer_436_vector_GA import ToricTransformer as Transformer_436, TransformerConfig

            config = TransformerConfig(self.device)
            self.model = Transformer_436(config)
            self.model.to(self.device)
            # load model
            self.model.load(PATH, optimizer=None)
            self.model = torch.compile(self.model)
            self.model.eval()
        else:
            raise ValueError("Input correct model type!")
    
    def predict(self, matrix):
        '''takes toric code matrix as numpy array and outputs predition as float'''
        if isinstance(matrix, np.ndarray):
            torch_matrix = torch.from_numpy(matrix)
        elif torch.is_tensor(matrix):
            torch_matrix = matrix
        else:
            raise TypeError("Input is neither numpy.ndarray, nor torch.Tensor")
        
        if self.model_type == "Transformer_v43":
            if len(torch_matrix.shape)<3:
                code = torch.unsqueeze(torch_matrix, dim=0) # add batch dimension
            length = torch.tensor([code.shape[1]], dtype=torch.long)

            classes_all = torch.arange((self.field_base-1)**2, dtype=global_dtype)
            logits, _ = self.model(code.long(), length)
            prob = F.softmax(logits, dim=1).cpu() # shape (B, n_output)
            return torch.sum(prob * classes_all).detach().numpy()
        else:
            ValueError("Input correct model type!")
    
    def evaluate_mse(self, dataset):
        losses = torch.zeros(len(dataset))
        with torch.no_grad():
            for i, data in enumerate(dataset):
                code, mindist = data
                out_mindist = self.model(code)

                losses[i] = self.criterion(self.denormalize(out_mindist), mindist).item()

        return losses.mean()