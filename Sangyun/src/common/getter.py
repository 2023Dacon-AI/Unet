import segmentation_models_pytorch as smp

def get_model(architecture, init_params):
    model_class = smp.__dict__[architecture]
    return model_class(**init_params)
