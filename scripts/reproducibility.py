import os, torch, numpy as np

def set_config(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    torch.use_deterministic_algorithms(True)

    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    
def split_indices(dataset, train_percent=0.7, val_percent=0.2, test_percent=0.1):
    n = len(dataset)
    indices = list(range(n))
    np.random.shuffle(indices)

    first_split = int(np.floor(train_percent * n)) 
    second_split = int(np.floor((train_percent + val_percent) * n))

    train_indices = indices[:first_split]
    val_indices = indices[first_split:second_split]
    test_indices = indices[second_split:]
    
    return train_indices, val_indices, test_indices
    

