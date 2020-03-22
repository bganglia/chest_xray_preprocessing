from torch.utils.data import random_split

def grouped_split(dataset, groups, lengths):
    unique_groups = groups.unique()
    selected_groups = unique_groups(dataset)
    out = [dataset[groups==subset] for subset in dataset]
    return out
