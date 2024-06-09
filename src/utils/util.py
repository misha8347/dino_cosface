import yaml

def load_yaml(filename: str):
    with open(filename, 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg
    