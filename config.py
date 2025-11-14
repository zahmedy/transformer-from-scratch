from pathlib import Path

def get_config():
    return {
            "batch_size": 8,
            "num_epochs": 20,
            "lr": 10**-4,
            "seq_len": 350,
            "d_model": 512,
            "datasource": 'opus_books',
            "lang_src": "en",
            "lang_tgt": "it",
            "model_folder": "weights",
            "model_basename": "tmodel_",
            "preload": "latest",
            "tokenizer_file": "tokenizer_{0}.json",
            "experiment_name": "runs/tmodel",
            "num_workers": 4
        }

def get_weights_file_path(config, epoch: str):
    model_folder = config['model_folder']
    model_basename = config.get('model_basename') #or config.get('model_filename')
    if model_basename is None:
        raise KeyError("Config must define 'model_basename' (preferred) or 'model_filename'")
    model_filename = f"{model_basename}{epoch}.pt"
    return str(Path('.') / model_folder / model_filename)
