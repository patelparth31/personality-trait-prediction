import pickle
import os

def load_models():
    models = {}
    models_directory = 'models'
    for filename in os.listdir(models_directory):
        if filename.endswith('.pkl'):
            trait = filename.split('_')[0]
            model_path = os.path.join(models_directory, filename)
            with open(model_path, 'rb') as file:
                models[trait] = pickle.load(file)
    return models
