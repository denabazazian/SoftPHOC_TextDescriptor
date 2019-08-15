from dataloaders.datasets import icdar
from dataloaders.datasets import synthText
#from dataloaders.datasets import icdar_sanity_check as icdar

from torch.utils.data import DataLoader

def make_data_loader(args, **kwargs):

    if args.dataset == 'icdar':
        train_set = icdar.ICDARSegmentation(args, split='train')
        val_set = icdar.ICDARSegmentation(args, split='val')

        num_class = train_set.NUM_CLASSES
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        test_loader = None

        return train_loader, val_loader, test_loader, num_class

    elif args.dataset == 'synthText':
        train_set = synthText.synthTextSegmentation(args, split='train')
        val_set = synthText.synthTextSegmentation(args, split='val')

        num_class = train_set.NUM_CLASSES
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        test_loader = None

        return train_loader, val_loader, test_loader, num_class

    else:
        raise NotImplementedError

