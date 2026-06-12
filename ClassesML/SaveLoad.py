import os
import torch

class SaveLoad:
    def __init__(self,path_model:str,model_name):
        self.path_model = path_model
        self.model_name = model_name
    
    def save_model(self,model):
        name = str(self.model_name)+'model.pth'
        save_path = os.path.join(self.path_model,''+name)
        torch.save(model.state_dict(),save_path)
        print(f"Model {name} saved at: {save_path}")
    
    def load_model(self,model):
        name = str(self.model_name) + "model.pth"
        save_path = os.path.join(self.path_model, "" + name)
        model.load_state_dict(torch.load(save_path))
        print(f"Model {name} loaded from: {save_path}")
        return model
    