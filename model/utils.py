def check_dim(dim, target):
    if isinstance(dim, int):
        dim = [dim] * target
    else:
        assert len(dim) == target
    return dim