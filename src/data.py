import copy
import torch
import numpy as np
import models
from config import cfg
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataloader import default_collate
from utils import collate, to_device
import copy
from datasets.voc import SimpleDataset
# import albumentations as A
# from albumentations.pytorch import ToTensorV2
import torchvision.transforms as T


data_stats = {'MNIST': ((0.1307,), (0.3081,)), 'FashionMNIST': ((0.2860,), (0.3530,)),
              'CIFAR10': ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
              'CIFAR100': ((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
              'SVHN': ((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)),
              'STL10': ((0.4409, 0.4279, 0.3868), (0.2683, 0.2610, 0.2687)),
              'voc' :((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))}

from datasets import VOCSegmentation
mask_path = 'data/masks.npy'
impath = 'data/images.npy'
def copy_dataset(ds):
    ds1 = VOCSegmentation(impath,mask_path)
    for i in dir(ds):
        if i[0] == '_':
            continue
        if i in ['data','target']:
            continue
        else:
            #i = 'data'
            expression = f"ds1.{i} = copy.deepcopy(ds.{i})"
            exec(expression)
    return ds1

def copy_dataset1():
    ds1 = SimpleDataset()
    for i in dir(ds):
        if i[0] == '_':
            continue
        if i in ['id','data','target']:
            continue
        else:
            #i = 'data'
            expression = f"ds1.{i} = ds1.{i}"
            exec(expression)
    return ds1


def fetch_dataset(data_name):
    import datasets
    dataset = {}
    print('fetching data {}...'.format(data_name))

    root = './data/{}'.format(data_name)
    if data_name in ['MNIST', 'FashionMNIST']:
        dataset['train'] = eval('datasets.{}(root=root, split=\'train\', '
                                'transform=datasets.Compose([transforms.ToTensor()]))'.format(data_name))
        dataset['test'] = eval('datasets.{}(root=root, split=\'test\', '
                               'transform=datasets.Compose([transforms.ToTensor()]))'.format(data_name))
        dataset['train'].transform = datasets.Compose([
            transforms.ToTensor(),
            transforms.Normalize(*data_stats[data_name])])
        dataset['test'].transform = datasets.Compose([
            transforms.ToTensor(),
            transforms.Normalize(*data_stats[data_name])])
    elif data_name in ['CIFAR10', 'CIFAR100']:
        dataset['train'] = eval('datasets.{}(root=root, split=\'train\', '
                                'transform=datasets.Compose([transforms.ToTensor()]))'.format(data_name))
        dataset['test'] = eval('datasets.{}(root=root, split=\'test\', '
                               'transform=datasets.Compose([transforms.ToTensor()]))'.format(data_name))
        dataset['train'].transform = datasets.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
            transforms.ToTensor(),
            transforms.Normalize(*data_stats[data_name])])
        dataset['test'].transform = datasets.Compose([
            transforms.ToTensor(),
            transforms.Normalize(*data_stats[data_name])])
    elif data_name in ['SVHN']:
        dataset['train'] = eval('datasets.{}(root=root, split=\'train\', '
                                'transform=datasets.Compose([transforms.ToTensor()]))'.format(data_name))
        dataset['test'] = eval('datasets.{}(root=root, split=\'test\', '
                               'transform=datasets.Compose([transforms.ToTensor()]))'.format(data_name))
        dataset['train'].transform = datasets.Compose([
            transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
            transforms.ToTensor(),
            transforms.Normalize(*data_stats[data_name])])
        dataset['test'].transform = datasets.Compose([
            transforms.ToTensor(),
            transforms.Normalize(*data_stats[data_name])])
    elif data_name in ['STL10']:
        dataset['train'] = eval('datasets.{}(root=root, split=\'train\', '
                                'transform=datasets.Compose([transforms.ToTensor()]))'.format(data_name))
        dataset['test'] = eval('datasets.{}(root=root, split=\'test\', '
                               'transform=datasets.Compose([transforms.ToTensor()]))'.format(data_name))
        dataset['train'].transform = datasets.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(96, padding=12, padding_mode='reflect'),
            transforms.ToTensor(),
            transforms.Normalize(*data_stats[data_name])])
        dataset['test'].transform = datasets.Compose([
            transforms.ToTensor(),
            transforms.Normalize(*data_stats[data_name])])
    elif data_name in ['voc']:
      mask_path = 'data/masks.npy'
      impath = 'data/images.npy'
      dataset['train'] = datasets.VOCSegmentation(images_path = impath,masks_path = mask_path,split = 'train')
      dataset['test'] = datasets.VOCSegmentation(images_path = impath,masks_path = mask_path,split = 'test')
        
    else:
        raise ValueError('Not valid dataset name')
    print('data ready')
    return dataset


def input_collate(batch):
    if isinstance(batch[0], dict):
        output = {key: [] for key in batch[0].keys()}
        for b in batch:
            for key in b:
                output[key].append(b[key])
        return output
    else:
        return default_collate(batch)


def make_data_loader(dataset, tag, batch_size=None, shuffle=None, sampler=None, batch_sampler=None):
    data_loader = {}
    for k in dataset:
        _batch_size = cfg[tag]['batch_size'][k] if batch_size is None else batch_size[k]
        _shuffle = cfg[tag]['shuffle'][k] if shuffle is None else shuffle[k]
        #print("shuffle  :",_shuffle)
        if cfg['data_name'] in ['voc']:
            _batch_size = 9
            #rs = torch.utils.data.RandomSampler(dataset[k], replacement=False, num_samples=None)
            data_loader[k] = DataLoader(dataset=dataset[k], batch_size=_batch_size, shuffle = _shuffle,
                                        pin_memory=True, num_workers=cfg['num_workers'],
                                        worker_init_fn=np.random.seed(cfg['seed']))
        elif sampler is not None:
            data_loader[k] = DataLoader(dataset=dataset[k], batch_size=_batch_size, sampler=sampler[k],
                                        pin_memory=True, num_workers=cfg['num_workers'], collate_fn=input_collate,
                                        worker_init_fn=np.random.seed(cfg['seed']))
        elif batch_sampler is not None:
            data_loader[k] = DataLoader(dataset=dataset[k], batch_sampler=batch_sampler[k],
                                        pin_memory=True, num_workers=cfg['num_workers'], collate_fn=input_collate,
                                        worker_init_fn=np.random.seed(cfg['seed']))
        else:
            data_loader[k] = DataLoader(dataset=dataset[k], batch_size=_batch_size, shuffle=_shuffle,
                                        pin_memory=True, num_workers=cfg['num_workers'], collate_fn=input_collate,
                                        worker_init_fn=np.random.seed(cfg['seed']))

    return data_loader


def split_dataset(dataset, num_users, data_split_mode):
    data_split = {}
    if data_split_mode == 'iid':
        data_split['train'] = iid(dataset['train'], num_users)
        data_split['test'] = iid(dataset['test'], num_users)
    elif 'non-iid' in cfg['data_split_mode']:
        data_split['train'] = non_iid(dataset['train'], num_users)
        data_split['test'] = non_iid(dataset['test'], num_users)
    else:
        raise ValueError('Not valid data split mode')
    return data_split


def iid(dataset, num_users):
    num_items = int(len(dataset) / num_users)
    data_split, idx = {}, list(range(len(dataset)))
    for i in range(num_users):
        num_items_i = min(len(idx), num_items)
        data_split[i] = torch.tensor(idx)[torch.randperm(len(idx))[:num_items_i]].tolist()
        idx = list(set(idx) - set(data_split[i]))
    return data_split


def non_iid(dataset, num_users):
    target = torch.tensor(dataset.target)
    data_split_mode_list = cfg['data_split_mode'].split('-')
    data_split_mode_tag = data_split_mode_list[-2]
    if data_split_mode_tag == 'l':
        data_split = {i: [] for i in range(num_users)}
        shard_per_user = int(data_split_mode_list[-1])
        target_idx_split = {}
        shard_per_class = int(shard_per_user * num_users / cfg['target_size'])
        for target_i in range(cfg['target_size']):
            target_idx = torch.where(target == target_i)[0]
            num_leftover = len(target_idx) % shard_per_class
            leftover = target_idx[-num_leftover:] if num_leftover > 0 else []
            new_target_idx = target_idx[:-num_leftover] if num_leftover > 0 else target_idx
            new_target_idx = new_target_idx.reshape((shard_per_class, -1)).tolist()
            for i, leftover_target_idx in enumerate(leftover):
                new_target_idx[i] = new_target_idx[i] + [leftover_target_idx.item()]
            target_idx_split[target_i] = new_target_idx
        target_split = list(range(cfg['target_size'])) * shard_per_class
        target_split = torch.tensor(target_split)[torch.randperm(len(target_split))].tolist()
        target_split = torch.tensor(target_split).reshape((num_users, -1)).tolist()
        for i in range(num_users):
            for target_i in target_split[i]:
                idx = torch.randint(len(target_idx_split[target_i]), (1,)).item()
                data_split[i].extend(target_idx_split[target_i].pop(idx))
    elif data_split_mode_tag == 'd':
        beta = float(data_split_mode_list[-1])
        dir = torch.distributions.dirichlet.Dirichlet(torch.tensor(beta).repeat(num_users))
        min_size = 0
        required_min_size = 10
        N = target.size(0)
        while min_size < required_min_size:
            data_split = [[] for _ in range(num_users)]
            for target_i in range(cfg['target_size']):
                target_idx = torch.where(target == target_i)[0]
                proportions = dir.sample()
                proportions = torch.tensor(
                    [p * (len(data_split_idx) < (N / num_users)) for p, data_split_idx in zip(proportions, data_split)])
                proportions = proportions / proportions.sum()
                split_idx = (torch.cumsum(proportions, dim=-1) * len(target_idx)).long().tolist()[:-1]
                split_idx = torch.tensor_split(target_idx, split_idx)
                data_split = [data_split_idx + idx.tolist() for data_split_idx, idx in zip(data_split, split_idx)]
            min_size = min([len(data_split_idx) for data_split_idx in data_split])
        data_split = {i: data_split[i] for i in range(num_users)}
    else:
        raise ValueError('Not valid data split mode tag')
    return data_split


def separate_dataset(dataset, idx):

    if cfg['data_name'] in ['voc']:
        # separated_dataset = SimpleDataset()
        # separated_dataset.target = torch.tensor(new_target.tolist())
        # separated_dataset.data = dataset.data
        # mix_dataset.ind = dataset.ind
        separated_dataset = copy_dataset(dataset)
        #separated_dataset = copy.deepcopy(dataset)
        separated_dataset.ind = np.array(idx)
        return separated_dataset
            
    separated_dataset = copy.deepcopy(dataset)
    separated_dataset.data = [dataset.data[s] for s in idx]
    separated_dataset.target = [dataset.target[s] for s in idx]
    # if cfg['data_name'] in ['voc']:
    #     separated_dataset.id = [dataset.id[s] for s in idx]
    #     separated_dataset.other['id'] = separated_dataset.id
    # else:
    separated_dataset.other['id'] = list(range(len(separated_dataset.data)))
    return separated_dataset


def separate_dataset_su(server_dataset, client_dataset=None, supervised_idx=None):
    if supervised_idx is None:
        if cfg['data_name'] in ['STL10']:
            if cfg['num_supervised'] == -1:
                supervised_idx = torch.arange(5000).tolist()
            else:
                target = torch.tensor(server_dataset.target)[:5000]
                num_supervised_per_class = cfg['num_supervised'] // cfg['target_size']
                supervised_idx = []
                for i in range(cfg['target_size']):
                    idx = torch.where(target == i)[0]
                    idx = idx[torch.randperm(len(idx))[:num_supervised_per_class]].tolist()
                    supervised_idx.extend(idx)
        elif cfg['data_name'] in ['voc']:
            supervised_idx = server_dataset.ind[np.arange(500)].tolist()

        else:
            if cfg['num_supervised'] == -1:
                supervised_idx = list(range(len(server_dataset)))
            else:
                target = torch.tensor(server_dataset.target)
                #target = server_dataset.mask.long()
                num_supervised_per_class = cfg['num_supervised'] // cfg['target_size']
                supervised_idx = []
                for i in range(cfg['target_size']):
                    idx = torch.where(target == i)[0]
                    idx = idx[torch.randperm(len(idx))[:num_supervised_per_class]].tolist()
                    supervised_idx.extend(idx)
    if cfg['data_name'] in ['voc']:
        idx = server_dataset.ind[list(range(len(server_dataset)))].tolist()
    else:
        idx = list(range(len(server_dataset)))
    unsupervised_idx = list(set(idx) - set(supervised_idx))
    _server_dataset = separate_dataset(server_dataset, supervised_idx)
    if client_dataset is None:
        _client_dataset = separate_dataset(server_dataset, unsupervised_idx)
    else:
        _client_dataset = separate_dataset(client_dataset, unsupervised_idx)
        transform = FixTransform(cfg['data_name'])
        _client_dataset.transform = transform
    # if cfg['data_name'] in ['voc']:
    #     _server_dataset.ind = np.array(supervised_idx)
    #     _client_dataset.ind = np.array(unsupervised_idx)

    return _server_dataset, _client_dataset, supervised_idx


def make_batchnorm_dataset_su(server_dataset, client_dataset):
    batchnorm_dataset = copy.deepcopy(server_dataset)
    batchnorm_dataset.data = batchnorm_dataset.data + client_dataset.data
    batchnorm_dataset.target = batchnorm_dataset.target + client_dataset.target
    batchnorm_dataset.other['id'] = batchnorm_dataset.other['id'] + client_dataset.other['id']
    return batchnorm_dataset


def make_dataset_normal(dataset):
    import datasets
    _transform = dataset.transform
    transform = datasets.Compose([transforms.ToTensor(), transforms.Normalize(*data_stats[cfg['data_name']])])
    dataset.transform = transform
    return dataset, _transform


def make_batchnorm_stats(dataset, model, tag):
    with torch.no_grad():
        test_model = copy.deepcopy(model)
        test_model.apply(lambda m: models.make_batchnorm(m, momentum=None, track_running_stats=True))
        #dataset, _transform = make_dataset_normal(dataset)
        data_loader = make_data_loader({'train': dataset}, tag, shuffle={'train': False})['train']
        test_model.train(True)
        for i, input in enumerate(data_loader):
            #input = collate(input)
            input = to_device(input, cfg['device'])
            test_model(input)
        #dataset.transform = _transform
    return test_model


class FixTransform(object):
    def __init__(self, data_name):
        import datasets
        self.segmentation = 0
        if data_name in ['CIFAR10', 'CIFAR100']:
            self.weak = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
                transforms.ToTensor(),
                transforms.Normalize(*data_stats[data_name])
            ])
            self.strong = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
                datasets.RandAugment(n=2, m=10),
                transforms.ToTensor(),
                transforms.Normalize(*data_stats[data_name])
            ])
        elif data_name in ['SVHN']:
            self.weak = transforms.Compose([
                transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
                transforms.ToTensor(),
                transforms.Normalize(*data_stats[data_name])
            ])
            self.strong = transforms.Compose([
                transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
                datasets.RandAugment(n=2, m=10),
                transforms.ToTensor(),
                transforms.Normalize(*data_stats[data_name])
            ])
        elif data_name in ['STL10']:
            self.weak = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(96, padding=12, padding_mode='reflect'),
                transforms.ToTensor(),
                transforms.Normalize(*data_stats[data_name])
            ])
            self.strong = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(96, padding=12, padding_mode='reflect'),
                datasets.RandAugment(n=2, m=10),
                transforms.ToTensor(),
                transforms.Normalize(*data_stats[data_name])
            ])
        elif data_name in ['voc']:
            self.segmentation = 1
            self.weak = torchvision.transforms.v2.GaussianBlur(kernel_size=(5,5), sigma=(0.1))
            self.strong =  torchvision.transforms.v2.Compose([
                                #transforms.ColorJitter(contrast=0.5),
                                 torchvision.transforms.v2.GaussianBlur(kernel_size=(5,5), sigma=(0.1,1.)),
                                 torchvision.transforms.v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
                                #transforms.ToTensor()
                                #transforms.CenterCrop(480),
                            ])
        else:
            raise ValueError('Not valid dataset')

    def __call__(self, input):
        # if self.segmentation:
        #   data,target = self.weak(input['data'],input['target'])
        #   aug,aug_target = self.strong(input['data'],input['target'])
        #   input = {**input, 'data':data,'target':target,}
        data = self.weak(input['data'])
        aug = self.strong(input['data'])
        
        input = {**input, 'data': data, 'aug': aug}
        return input

class MixDataset(Dataset):
    def __init__(self, size, dataset):
        self.size = size
        self.dataset = dataset

    def __getitem__(self, index):
        index = torch.randint(0, len(self.dataset), (1,)).item()
        input = self.dataset[index]
        input = {'data': input['data'], 'target': input['target']}
        return input

    def __len__(self):
        return self.size


