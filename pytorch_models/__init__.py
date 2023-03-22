from pytorch_models.dcgan import BaseDiscriminator, BaseGenerator
from pytorch_models.dccgan import CondDiscriminator, CondGenerator
from pytorch_models.rccgan import ResDiscriminator, ResGenerator

# Path: Final Project/pytorch_models/__init__.py

class GANFactory():
    '''
    '''
    
    def get_model(self, model_type: str, params: dict):
        device = params['device']
        if model_type == 'DC-GAN':
            return BaseDiscriminator(params).to(device), BaseGenerator(params).to(device)
        elif model_type == 'DC-cGAN':
            return CondDiscriminator(params).to(device), CondGenerator(params).to(device)
        elif model_type == 'RC-cGAN':
            return ResDiscriminator(params).to(device), ResGenerator(params).to(device)
        else:
            raise ValueError('Invalid model type')