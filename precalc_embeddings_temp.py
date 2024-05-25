import io
import pickle

import torch

class _CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else: return super().find_class(module, name)


class GraphLoader:
    @staticmethod
    def load_graph(load_path):
        with open(load_path, 'rb') as file:
            return _CPU_Unpickler(file).load()
