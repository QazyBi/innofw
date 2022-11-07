from torch.optim import SGD as Sgd, Adam


class SGD:
    optimizer = Sgd


class ADAM:
    """
    Class defines a wrapper of the torch optimizer to illustrate
    how to use custom optimizer implementations in the innofw
    Attributes
    ----------
    optimizer : torch.optim
        optimizer from torch framework
    """
    optimizer = Adam
