import os
import gdown
import zipfile
import torch
import numpy as np
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

from torch.utils.data import DataLoader
import torchvision as tvs
if tvs.__version__ >= '0.13':
    tvs_new = True
else:
    tvs_new = False

from openood.datasets.imglist_dataset import ImglistDataset
from openood.preprocessors import BasePreprocessor
from openood.evaluation_api.color_mnist import get_biased_mnist_dataloader, get_biased_mnist_dataset
from openood.evaluation_api.celebA_dataset import get_celebA_dataloader, celebAOodDataset
from openood.evaluation_api.cub_dataset import get_waterbird_dataloader
from openood.utils.svhn_loader import SVHN
import openood.utils.svhn_loader as svhn

import argparse
from .preprocessor import get_default_preprocessor, ImageNetCPreProcessor

np.random.seed(0)
torch.manual_seed(0)

DATA_INFO = {
    'cifar10': {
        'num_classes': 10,
        'id': {
            'train': {
                'data_dir': 'images_classic/',
                'imglist_path': 'benchmark_imglist/cifar10/train_cifar10.txt'
            },
            'val': {
                'data_dir': 'images_classic/',
                'imglist_path': 'benchmark_imglist/cifar10/val_cifar10.txt'
            },
            'test': {
                'data_dir': 'images_classic/',
                'imglist_path': 'benchmark_imglist/cifar10/test_cifar10.txt'
            }
        },
        'csid': {
            'datasets': ['cifar10c'],
            'cinic10': {
                'data_dir': 'images_classic/',
                'imglist_path': 'benchmark_imglist/cifar10/val_cinic10.txt'
            },
            'cifar10c': {
                'data_dir': 'images_classic/',
                'imglist_path': 'benchmark_imglist/cifar10/test_cifar10c.txt'
            }
        },
        'ood': {
            'val': {
                'data_dir': 'images_classic/',
                'imglist_path': 'benchmark_imglist/cifar10/val_tin.txt'
            },
            'near': {
                'datasets': ['cifar100', 'tin'],
                'cifar100': {
                    'data_dir': 'images_classic/',
                    'imglist_path':
                    'benchmark_imglist/cifar10/test_cifar100.txt'
                },
                'tin': {
                    'data_dir': 'images_classic/',
                    'imglist_path': 'benchmark_imglist/cifar10/test_tin.txt'
                }
            },
            'far': {
                'datasets': ['mnist', 'svhn', 'texture', 'places365'],
                'mnist': {
                    'data_dir': 'images_classic/',
                    'imglist_path': 'benchmark_imglist/cifar10/test_mnist.txt'
                },
                'svhn': {
                    'data_dir': 'images_classic/',
                    'imglist_path': 'benchmark_imglist/cifar10/test_svhn.txt'
                },
                'texture': {
                    'data_dir': 'images_classic/',
                    'imglist_path':
                    'benchmark_imglist/cifar10/test_texture.txt'
                },
                'places365': {
                    'data_dir': 'images_classic/',
                    'imglist_path':
                    'benchmark_imglist/cifar10/test_places365.txt'
                },
            }
        }
    },
    'waterbirds': {
        'num_classes': 2,
        'id': {
            'train': {
                'data_dir': 'images_classic/',
                'imglist_path': 'benchmark_imglist/waterbirds/train_waterbirds.txt'
            },
            'val': {
                'data_dir': 'images_classic/',
                'imglist_path': 'benchmark_imglist/waterbirds/val_waterbirds.txt'
            },
            'test': {
                'data_dir': 'images_classic/',
                'imglist_path': 'benchmark_imglist/waterbirds/test_waterbirds.txt'
            }
        },
        'csid': {
            'datasets': [],
        },
        'ood': {
            'val': {
                'data_dir': 'images_classic/',
                'imglist_path': 'benchmark_imglist/waterbirds/val_tin.txt'
            },
            'near': {
                'datasets': ['bg', 'sd'],
                'bg': {
                    'data_dir': 'images_classic/',
                    'imglist_path':
                    'benchmark_imglist/waterbirds/test_bg.txt'
                },
                'sd': {
                    'data_dir': 'images_classic/',
                    'imglist_path': 'benchmark_imglist/waterbirds/test_sd.txt'
                }
            },
            'far': {
                'datasets': ['mnist', 'svhn', 'texture', 'places365'],
                'mnist': {
                    'data_dir': 'images_classic/',
                    'imglist_path': 'benchmark_imglist/waterbirds/test_mnist.txt'
                },
                'svhn': {
                    'data_dir': 'images_classic/',
                    'imglist_path': 'benchmark_imglist/waterbirds/test_svhn.txt'
                },
                'texture': {
                    'data_dir': 'images_classic/',
                    'imglist_path':
                    'benchmark_imglist/waterbirds/test_texture.txt'
                },
                'places365': {
                    'data_dir': 'images_classic/',
                    'imglist_path':
                    'benchmark_imglist/waterbirds/test_places365.txt'
                }
            },
        }
    },

    'celebA': {
        'num_classes': 2,
        'id': {
            'train': {
                'data_dir': 'images_classic/',
                'imglist_path': 'benchmark_imglist/celebA/train_celebA.txt'
            },
            'val': {
                'data_dir': 'images_classic/',
                'imglist_path': 'benchmark_imglist/celebA/val_celebA.txt'
            },
            'test': {
                'data_dir': 'images_classic/',
                'imglist_path': 'benchmark_imglist/celebA/test_celebA.txt'
            }
        },
        'csid': {
            'datasets': [],
        },
        'ood': {
            'val': {
                'data_dir': 'images_classic/',
                'imglist_path': 'benchmark_imglist/celebA/val_tin.txt'
            },
            'near': {
                'datasets': ['sp_celebA'],
                'sp_celebA': {
                    'data_dir': 'images_classic/',
                    'imglist_path':
                    'benchmark_imglist/celebA/test_sp_celebA.txt'
                }
            },
            'far': {
                'datasets': ['mnist', 'svhn', 'texture', 'places365'],
                'mnist': {
                    'data_dir': 'images_classic/',
                    'imglist_path': 'benchmark_imglist/celebA/test_mnist.txt'
                },
                'svhn': {
                    'data_dir': 'images_classic/',
                    'imglist_path': 'benchmark_imglist/celebA/test_svhn.txt'
                },
                'texture': {
                    'data_dir': 'images_classic/',
                    'imglist_path':
                    'benchmark_imglist/celebA/test_texture.txt'
                },
                'places365': {
                    'data_dir': 'images_classic/',
                    'imglist_path':
                    'benchmark_imglist/celebA/test_places365.txt'
                }
            },
        }
    },

     'cmnist': {
        'num_classes': 2,
        'id': {
            'train': {
                'data_dir': 'images_classic/',
                'imglist_path': 'benchmark_imglist/cmnist/train_cmnist.txt'
            },
            'val': {
                'data_dir': 'images_classic/',
                'imglist_path': 'benchmark_imglist/cmnist/val_cmnist.txt'
            },
            'test': {
                'data_dir': 'images_classic/',
                'imglist_path': 'benchmark_imglist/cmnist/test_cmnist.txt'
            }
        },
        'csid': {
            'datasets': [],
        },
        'ood': {
            'val': {
                'data_dir': 'images_classic/',
                'imglist_path': 'benchmark_imglist/cmnist/val_tin.txt'
            },
            'near': {
                'datasets': ['sp_cmnist'],
                'sp_cmnist': {
                    'data_dir': 'images_classic/',
                    'imglist_path':
                    'benchmark_imglist/cmnist/test_sp_cmnist.txt'
                }
            },
            'far': {
                'datasets': ['mnist', 'svhn', 'texture', 'places365'],
                'mnist': {
                    'data_dir': 'images_classic/',
                    'imglist_path': 'benchmark_imglist/cmnist/test_mnist.txt'
                },
                'svhn': {
                    'data_dir': 'images_classic/',
                    'imglist_path': 'benchmark_imglist/cmnist/test_svhn.txt'
                },
                'texture': {
                    'data_dir': 'images_classic/',
                    'imglist_path':
                    'benchmark_imglist/cmnist/test_texture.txt'
                },
                'places365': {
                    'data_dir': 'images_classic/',
                    'imglist_path':
                    'benchmark_imglist/cmnist/test_places365.txt'
                }
            },
        }
    },

      
    'cifar100': {
        'num_classes': 100,
        'id': {
            'train': {
                'data_dir': 'images_classic/',
                'imglist_path': 'benchmark_imglist/cifar100/train_cifar100.txt'
            },
            'val': {
                'data_dir': 'images_classic/',
                'imglist_path': 'benchmark_imglist/cifar100/val_cifar100.txt'
            },
            'test': {
                'data_dir': 'images_classic/',
                'imglist_path': 'benchmark_imglist/cifar100/test_cifar100.txt'
            }
        },
        'csid': {
            'datasets': [],
        },
        'ood': {
            'val': {
                'data_dir': 'images_classic/',
                'imglist_path': 'benchmark_imglist/cifar100/val_tin.txt'
            },
            'near': {
                'datasets': ['cifar10', 'tin'],
                'cifar10': {
                    'data_dir': 'images_classic/',
                    'imglist_path':
                    'benchmark_imglist/cifar100/test_cifar10.txt'
                },
                'tin': {
                    'data_dir': 'images_classic/',
                    'imglist_path': 'benchmark_imglist/cifar100/test_tin.txt'
                }
            },
            'far': {
                'datasets': ['mnist', 'svhn', 'texture', 'places365'],
                'mnist': {
                    'data_dir': 'images_classic/',
                    'imglist_path': 'benchmark_imglist/cifar100/test_mnist.txt'
                },
                'svhn': {
                    'data_dir': 'images_classic/',
                    'imglist_path': 'benchmark_imglist/cifar100/test_svhn.txt'
                },
                'texture': {
                    'data_dir': 'images_classic/',
                    'imglist_path':
                    'benchmark_imglist/cifar100/test_texture.txt'
                },
                'places365': {
                    'data_dir': 'images_classic/',
                    'imglist_path':
                    'benchmark_imglist/cifar100/test_places365.txt'
                }
            },
        }
    },


    
    'imagenet200': {
        'num_classes': 200,
        'id': {
            'train': {
                'data_dir':
                'images_largescale/',
                'imglist_path':
                'benchmark_imglist/imagenet200/train_imagenet200.txt'
            },
            'val': {
                'data_dir': 'images_largescale/',
                'imglist_path':
                'benchmark_imglist/imagenet200/val_imagenet200.txt'
            },
            'test': {
                'data_dir':
                'images_largescale/',
                'imglist_path':
                'benchmark_imglist/imagenet200/test_imagenet200.txt'
            }
        },
        'csid': {
            'datasets': ['imagenet_v2', 'imagenet_c', 'imagenet_r'],
            'imagenet_v2': {
                'data_dir':
                'images_largescale/',
                'imglist_path':
                'benchmark_imglist/imagenet200/test_imagenet200_v2.txt'
            },
            'imagenet_c': {
                'data_dir':
                'images_largescale/',
                'imglist_path':
                'benchmark_imglist/imagenet200/test_imagenet200_c.txt'
            },
            'imagenet_r': {
                'data_dir':
                'images_largescale/',
                'imglist_path':
                'benchmark_imglist/imagenet200/test_imagenet200_r.txt'
            },
        },
        'ood': {
            'val': {
                'data_dir': 'images_largescale/',
                'imglist_path':
                'benchmark_imglist/imagenet200/val_openimage_o.txt'
            },
            'near': {
                'datasets': ['ssb_hard', 'ninco'],
                'ssb_hard': {
                    'data_dir':
                    'images_largescale/',
                    'imglist_path':
                    'benchmark_imglist/imagenet200/test_ssb_hard.txt'
                },
                'ninco': {
                    'data_dir': 'images_largescale/',
                    'imglist_path':
                    'benchmark_imglist/imagenet200/test_ninco.txt'
                }
            },
            'far': {
                'datasets': ['inaturalist', 'textures', 'openimage_o'],
                'inaturalist': {
                    'data_dir':
                    'images_largescale/',
                    'imglist_path':
                    'benchmark_imglist/imagenet200/test_inaturalist.txt'
                },
                'textures': {
                    'data_dir':
                    'images_classic/',
                    'imglist_path':
                    'benchmark_imglist/imagenet200/test_textures.txt'
                },
                'openimage_o': {
                    'data_dir':
                    'images_largescale/',
                    'imglist_path':
                    'benchmark_imglist/imagenet200/test_openimage_o.txt'
                },
            },
        }
    },
    'imagenet': {
        'num_classes': 1000,
        'id': {
            'train': {
                'data_dir': 'images_largescale/',
                'imglist_path': 'benchmark_imglist/imagenet/train_imagenet.txt'
            },
            'val': {
                'data_dir': 'images_largescale/',
                'imglist_path': 'benchmark_imglist/imagenet/val_imagenet.txt'
            },
            'test': {
                'data_dir': 'images_largescale/',
                'imglist_path': 'benchmark_imglist/imagenet/test_imagenet.txt'
            }
        },
        'csid': {
            'datasets': ['imagenet_v2', 'imagenet_c', 'imagenet_r'],
            'imagenet_v2': {
                'data_dir': 'images_largescale/',
                'imglist_path':
                'benchmark_imglist/imagenet/test_imagenet_v2.txt'
            },
            'imagenet_c': {
                'data_dir': 'images_largescale/',
                'imglist_path':
                'benchmark_imglist/imagenet/test_imagenet_c.txt'
            },
            'imagenet_r': {
                'data_dir': 'images_largescale/',
                'imglist_path':
                'benchmark_imglist/imagenet/test_imagenet_r.txt'
            },
        },
        'ood': {
            'val': {
                'data_dir': 'images_largescale/',
                'imglist_path':
                'benchmark_imglist/imagenet/val_openimage_o.txt'
            },
            'near': {
                'datasets': ['ssb_hard', 'ninco'],
                'ssb_hard': {
                    'data_dir': 'images_largescale/',
                    'imglist_path':
                    'benchmark_imglist/imagenet/test_ssb_hard.txt'
                },
                'ninco': {
                    'data_dir': 'images_largescale/',
                    'imglist_path': 'benchmark_imglist/imagenet/test_ninco.txt'
                }
            },
            'far': {
                'datasets': ['inaturalist', 'textures', 'openimage_o'],
                'inaturalist': {
                    'data_dir':
                    'images_largescale/',
                    'imglist_path':
                    'benchmark_imglist/imagenet/test_inaturalist.txt'
                },
                'textures': {
                    'data_dir': 'images_classic/',
                    'imglist_path':
                    'benchmark_imglist/imagenet/test_textures.txt'
                },
                'openimage_o': {
                    'data_dir':
                    'images_largescale/',
                    'imglist_path':
                    'benchmark_imglist/imagenet/test_openimage_o.txt'
                },
            },
        }
    },
}

download_id_dict = {
    'cifar10': '1Co32RiiWe16lTaiOU6JMMnyUYS41IlO1',
    'cifar100': '1PGKheHUsf29leJPPGuXqzLBMwl8qMF8_',
    'tin': '1PZ-ixyx52U989IKsMA2OT-24fToTrelC',
    'mnist': '1CCHAGWqA1KJTFFswuF9cbhmB-j98Y1Sb',
    'svhn': '1DQfc11HOtB1nEwqS4pWUFp8vtQ3DczvI',
    'texture': '1OSz1m3hHfVWbRdmMwKbUzoU8Hg9UKcam',
    'places365': '1Ec-LRSTf6u5vEctKX9vRp9OA6tqnJ0Ay',
    'imagenet_1k': '1i1ipLDFARR-JZ9argXd2-0a6DXwVhXEj',
    'species_sub': '1-JCxDx__iFMExkYRMylnGJYTPvyuX6aq',
    'ssb_hard': '1PzkA-WGG8Z18h0ooL_pDdz9cO-DCIouE',
    'ninco': '1Z82cmvIB0eghTehxOGP5VTdLt7OD3nk6',
    'inaturalist': '1zfLfMvoUD0CUlKNnkk7LgxZZBnTBipdj',
    'places': '1fZ8TbPC4JGqUCm-VtvrmkYxqRNp2PoB3',
    'sun': '1ISK0STxWzWmg-_uUr4RQ8GSLFW7TZiKp',
    'openimage_o': '1VUFXnB_z70uHfdgJG2E_pjYOcEgqM7tE',
    'imagenet_v2': '1akg2IiE22HcbvTBpwXQoD7tgfPCdkoho',
    'imagenet_r': '1EzjMN2gq-bVV7lg-MEAdeuBuz-7jbGYU',
    'imagenet_c': '1JeXL9YH4BO8gCJ631c5BHbaSsl-lekHt',
    'benchmark_imglist': '1XKzBdWCqg3vPoj-D32YixJyJJ0hL63gP'
}

dir_dict = {
    'images_classic/': [
        'cifar100', 'tin', 'tin597', 'svhn', 'cinic10', 'imagenet10', 'mnist',
        'fashionmnist', 'cifar10', 'cifar100c', 'places365', 'cifar10c',
        'fractals_and_fvis', 'usps', 'texture', 'notmnist', 'waterbirds', 'celebA', 'cmnist'
    ],
    'images_largescale/': [
        'imagenet_1k',
        'ssb_hard',
        'ninco',
        'inaturalist',
        'places',
        'sun',
        'openimage_o',
        'imagenet_v2',
        'imagenet_c',
        'imagenet_r',
    ],
    'images_medical/': ['actmed', 'bimcv', 'ct', 'hannover', 'xraybone'],
}

benchmarks_dict = {
    'cifar10':
    ['cifar10', 'cifar100', 'tin', 'mnist', 'svhn', 'texture', 'places365'],
    'cifar100':
    ['cifar100', 'cifar10', 'tin', 'mnist', 'svhn', 'texture', 'places365'],
    'imagenet200': [
        'imagenet_1k', 'ssb_hard', 'ninco', 'inaturalist', 'texture',
        'openimage_o', 'imagenet_v2', 'imagenet_c', 'imagenet_r'
    ],
    'imagenet': [
        'imagenet_1k', 'ssb_hard', 'ninco', 'inaturalist', 'texture',
        'openimage_o', 'imagenet_v2', 'imagenet_c', 'imagenet_r'
    ],
    'waterbird':
    ['waterbirds', 'bg', 'sd', 'mnist', 'svhn', 'texture', 'places365'],
    'celebA':
    ['celebA', 'sp_celebA', 'mnist', 'svhn', 'texture', 'places365'],
    'cmnist':
    ['cmnist', 'sp_cmnist', 'mnist', 'svhn', 'texture', 'places365']
}


class GaussianDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_size, img_size = 32, labels = None, transform = None, num_classes = 2):
            if labels == None:
                self.labels = (torch.ones(dataset_size) * num_classes).long()
            else:
                self.labels = labels
#<<<<<<< HEAD
            torch.manual_seed(0)
#=======
#>>>>>>> origin/private-branch
            images = torch.normal(0.5, 0.5, size=(dataset_size,3,img_size,img_size))
            self.images = torch.clamp(images, 0, 1)
            self.transform = transform

    def __len__(self):
            return len(self.images)

    def __getitem__(self, index):
            # Load data and get label
            if self.transform:
                X = self.transform(self.images[index])
            else:
                X = self.images[index]
            y = self.labels[index]

            return X, y


def require_download(filename, path):
    for item in os.listdir(path):
        if item.startswith(filename) or filename.startswith(
                item) or path.endswith(filename):
            return False

    else:
        print(filename + ' needs download:')
        return True


def download_dataset(dataset, data_root):
    for key in dir_dict.keys():
        if dataset in dir_dict[key]:
            store_path = os.path.join(data_root, key, dataset)
            if not os.path.exists(store_path):
                os.makedirs(store_path)
            break
    else:
        print('Invalid dataset detected {}'.format(dataset))
        return

    if require_download(dataset, store_path):
        print(store_path)
        if not store_path.endswith('/'):
            store_path = store_path + '/'
        gdown.download(id=download_id_dict[dataset], output=store_path)

        file_path = os.path.join(store_path, dataset + '.zip')
        with zipfile.ZipFile(file_path, 'r') as zip_file:
            zip_file.extractall(store_path)
        os.remove(file_path)


def data_setup(data_root, id_data_name):
    if not data_root.endswith('/'):
        data_root = data_root + '/'

    if not os.path.exists(os.path.join(data_root, 'benchmark_imglist')):
        gdown.download(id=download_id_dict['benchmark_imglist'],
                       output=data_root)
        file_path = os.path.join(data_root, 'benchmark_imglist.zip')
        with zipfile.ZipFile(file_path, 'r') as zip_file:
            zip_file.extractall(data_root)
        os.remove(file_path)

    for dataset in benchmarks_dict[id_data_name]:
        download_dataset(dataset, data_root)


def get_id_ood_dataloader(id_name, data_root, preprocessor, **loader_kwargs):
    if 'cmnist' in id_name:
        dataloader_dict = {}
        sub_dataloader_dict = {}
        train_set1 = get_biased_mnist_dataset(root = './datasets/MNIST', batch_size=64,
                                        data_label_correlation= 0.45,
                                        n_confusing_labels= 1,
                                        train=True, partial=True, cmap = "1")
        train_set2 = get_biased_mnist_dataset(root = './datasets/MNIST', batch_size=64,
                                        data_label_correlation= 0.45,
                                        n_confusing_labels= 1,
                                        train=True, partial=True, cmap = "2")
        trainset = [train_set2, train_set1]
        kwargs = {'pin_memory': False, 'num_workers': 8, 'drop_last': True}


        sub_dataloader_dict['train'] = torch.utils.data.DataLoader(dataset=torch.utils.data.ConcatDataset(trainset), batch_size=64, shuffle=True, **kwargs)
        dataset_loader = get_biased_mnist_dataloader(root='./datasets/MNIST', batch_size=64,
                                              data_label_correlation=0.45,
                                              n_confusing_labels=1,
                                              train=False, partial=True, cmap="1")
        
        dataset_size = len(dataset_loader.dataset)
        indices = list(range(dataset_size))
        
        np.random.seed(0)
        np.random.shuffle(indices)
        
        split_index = int(dataset_size * 0.5)  
        val_indices, test_indices = indices[:split_index], indices[split_index:]

        val_dataset = torch.utils.data.Subset(dataset_loader.dataset, val_indices)
        test_dataset = torch.utils.data.Subset(dataset_loader.dataset, test_indices)

        sub_dataloader_dict['val'] = torch.utils.data.DataLoader(dataset=val_dataset,
                                                                batch_size=64, shuffle=False, **kwargs)
        sub_dataloader_dict['test'] = torch.utils.data.DataLoader(dataset=test_dataset,
                                                                 batch_size=64, shuffle=False, **kwargs)

        dataloader_dict['id'] = sub_dataloader_dict
        dataloader_dict['ood'] = {}

        small_transform = transforms.Compose([
                transforms.Resize((28,28)),
                transforms.CenterCrop(28),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, 0.5, 0.5),
                             std=(0.5, 0.5, 0.5))])
        testsetout = ImageFolder("/home/user01/SP_OOD_Experiments/OOD_Datasets/partial_color_mnist_0&1",
                                            transform=small_transform)
        np.random.seed(0)
        subset_indices = np.random.choice(len(testsetout), 2000, replace=False)
        subset_val_indices = subset_indices[:int(0.2 * len(subset_indices))]  
        subset_near_indices = subset_indices[int(0.2 * len(subset_indices)):] 

        subset_val = torch.utils.data.Subset(testsetout, subset_val_indices)
        sub_dataloader_dict_val = {'sp_cmnist': torch.utils.data.DataLoader(subset_val, batch_size=64, shuffle=False, num_workers=4)}
        dataloader_dict['ood']['val']= sub_dataloader_dict_val

        subset_near = torch.utils.data.Subset(testsetout, subset_near_indices)
        sub_dataloader_dict_near = {'sp_cmnist': torch.utils.data.DataLoader(subset_near, batch_size=64, shuffle=False, num_workers=4)}
        dataloader_dict['ood']['near'] = sub_dataloader_dict_near

        ood_datasets = ['textures', 'gaussian','LSUN_resize','iSUN']

        dataloader_dict['ood'].setdefault('far', {})
        dataloader_dict['ood'].setdefault('val', {})

        for dataset_name in ood_datasets:
            if dataset_name == 'gaussian':
                testsetout = GaussianDataset(dataset_size=10000, img_size=28,
                                            transform=transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
            else:
                testsetout = ImageFolder(f"/home/user01/SP_OOD_Experiments/OOD_Datasets/{dataset_name}", transform=small_transform)

            num_samples = len(testsetout)
            val_size = int(0.2 * num_samples) 
            np.random.seed(0) 
            val_indices = np.random.choice(num_samples, val_size, replace=False)
            np.random.seed(0)
            far_indices = np.random.choice(list(set(range(num_samples)) - set(val_indices)), num_samples - val_size, replace=False)

            subset_val = torch.utils.data.Subset(testsetout, val_indices)
            subset_far = torch.utils.data.Subset(testsetout, far_indices)
            sub_dataloader_dict = {}
            sub_dataloader_dict[dataset_name] = torch.utils.data.DataLoader(subset_far, batch_size=64, shuffle=False,
                                                                            num_workers=4)
            dataloader_dict['ood']['far'].update(sub_dataloader_dict)
            sub_dataloader_dict_val = {'sp_cmnist': sub_dataloader_dict_val['sp_cmnist']}
            sub_dataloader_dict_val[dataset_name] = torch.utils.data.DataLoader(subset_val, batch_size=64, shuffle=False,
                                                                                num_workers=4)
            dataloader_dict['ood']['val'].update(sub_dataloader_dict_val)
        return dataloader_dict
    
    elif 'waterbirds' in id_name:

        dataloader_dict = {}
        sub_dataloader_dict = {}
        train_loader = get_waterbird_dataloader( data_label_correlation=0.9, split="train")
        val_loader = get_waterbird_dataloader( data_label_correlation=0.9, split="val")


        sub_dataloader_dict['train'] = train_loader
        
        dataset_size = len(val_loader.dataset)
        indices = list(range(dataset_size))
        np.random.seed(0)
        np.random.shuffle(indices)
        
        split_index = int(dataset_size * 0.5)  
        val_indices, test_indices = indices[:split_index], indices[split_index:]

        val_dataset = torch.utils.data.Subset(val_loader.dataset, val_indices)
        test_dataset = torch.utils.data.Subset(val_loader.dataset, test_indices)

        kwargs = {'pin_memory': False, 'num_workers': 8, 'drop_last': True}

        sub_dataloader_dict['val'] = torch.utils.data.DataLoader(dataset=val_dataset,
                                                                batch_size=64, shuffle=False, **kwargs)
        sub_dataloader_dict['test'] = torch.utils.data.DataLoader(dataset=test_dataset,
                                                                 batch_size=64, shuffle=False, **kwargs)

        dataloader_dict['id'] = sub_dataloader_dict
        dataloader_dict['ood'] = {}
        
        scale = 256.0/224.0
        target_resolution = (224, 224)
        large_transform = transforms.Compose([
            transforms.Resize((int(target_resolution[0]*scale), int(target_resolution[1]*scale))),
            transforms.CenterCrop(target_resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        testsetout = ImageFolder("/content/drive/MyDrive/OOD_Datasets/placesbg/",transform=large_transform)
        np.random.seed(0)
        subset_indices = np.random.choice(len(testsetout), 2000, replace=False)
        subset_val_indices = subset_indices[:int(0.2 * len(subset_indices))]  
        subset_near_indices = subset_indices[int(0.2 * len(subset_indices)):]  
        subset_val = torch.utils.data.Subset(testsetout, subset_val_indices)
        # sub_dataloader_dict_val = {'sp_waterbirds': torch.utils.data.DataLoader(subset_val, batch_size=64, shuffle=False, num_workers=4)}
        # dataloader_dict['ood']['val']= sub_dataloader_dict_val

        testsetout2 = ImageFolder("/content/drive/MyDrive/OOD_Datasets/placesbg_diffusion/placesbg",transform=large_transform)
        np.random.seed(0)

        subset_indices2 = np.random.choice(len(testsetout2), 2000, replace=False)
        subset_val_indices2 = subset_indices2[:int(0.2 * len(subset_indices2))]  
        subset_near_indices2 = subset_indices2[int(0.2 * len(subset_indices2)):]  
        subset_val2 = torch.utils.data.Subset(testsetout2, subset_val_indices2)
        sub_dataloader_dict_val = {'sp_waterbirds': torch.utils.data.DataLoader(subset_val, batch_size=64, shuffle=False, num_workers=4),'stable_diffusion': torch.utils.data.DataLoader(subset_val2, batch_size=64, shuffle=False, num_workers=4)}
        dataloader_dict['ood']['val']= sub_dataloader_dict_val

        subset_near = torch.utils.data.Subset(testsetout, subset_near_indices)
        subset_near2 = torch.utils.data.Subset(testsetout2, subset_near_indices2)
        sub_dataloader_dict_near = {'sp_waterbirds': torch.utils.data.DataLoader(subset_near, batch_size=64, shuffle=False, num_workers=4),'stable_diffusion': torch.utils.data.DataLoader(subset_near2, batch_size=64, shuffle=False, num_workers=4)}
        dataloader_dict['ood']['near'] = sub_dataloader_dict_near


        ood_datasets = [ 'gaussian', 'SVHN', 'iSUN', 'LSUN_resize', 'textures']

        dataloader_dict['ood'].setdefault('far', {})
        dataloader_dict['ood'].setdefault('val', {})

        for dataset_name in ood_datasets:
            if dataset_name == "SVHN":
                testsetout = svhn.SVHN(f"/content/drive/MyDrive/OOD_Datasets/{dataset_name}", split='test',
                                    transform=large_transform, download=False)
            elif dataset_name == 'gaussian':
                testsetout = GaussianDataset(dataset_size =10000, img_size = 224,
                    transform=transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
            else:
                testsetout = ImageFolder(f"/content/drive/MyDrive/OOD_Datasets/{dataset_name}", transform=large_transform)

            num_samples = len(testsetout)
            val_size = int(0.2 * num_samples)  
            np.random.seed(0)
            val_indices = np.random.choice(num_samples, val_size, replace=False)
            np.random.seed(0)
            far_indices = np.random.choice(list(set(range(num_samples)) - set(val_indices)), num_samples - val_size, replace=False)

            # Create the 'far' subset for this dataset (80% - val_size)
            subset_val = torch.utils.data.Subset(testsetout, val_indices)
            subset_far = torch.utils.data.Subset(testsetout, far_indices)
            sub_dataloader_dict = {}
            sub_dataloader_dict[dataset_name] = torch.utils.data.DataLoader(subset_far, batch_size=64, shuffle=False,
                                                                            num_workers=4)
            dataloader_dict['ood']['far'].update(sub_dataloader_dict)
            # sub_dataloader_dict_val = {'sp_waterbirds': sub_dataloader_dict_val['sp_waterbirds']}
            sub_dataloader_dict_val[dataset_name] = torch.utils.data.DataLoader(subset_val, batch_size=64, shuffle=False ,
                                                                                num_workers=4)
            dataloader_dict['ood']['val'].update(sub_dataloader_dict_val)
        return dataloader_dict

    elif 'celebA' in id_name:
        dataloader_dict = {}
        sub_dataloader_dict = {}
        train_loader = get_celebA_dataloader(split="train")
        val_loader = get_celebA_dataloader(split="val")
        test_loader = get_celebA_dataloader(split="test")

        sub_dataloader_dict['train'] = train_loader
        sub_dataloader_dict['val'] = val_loader
        sub_dataloader_dict['test'] = test_loader

        dataloader_dict['id'] = sub_dataloader_dict
        dataloader_dict['ood'] = {}
        
        scale = 256.0/224.0
        target_resolution = (224, 224)
        large_transform = transforms.Compose([
            transforms.Resize((int(target_resolution[0]*scale), int(target_resolution[1]*scale))),
            transforms.CenterCrop(target_resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])


        ood_datasets = ['celebA_ood', 'gaussian', 'SVHN', 'iSUN', 'LSUN_resize']
        dataloader_dict['ood'].setdefault('near', {})
        dataloader_dict['ood'].setdefault('far', {})
        dataloader_dict['ood'].setdefault('val', {})

        for out_dataset in ood_datasets:
            if out_dataset == "SVHN":
                testsetout = svhn.SVHN(f"/content/drive/MyDrive/SP_OOD_Experiments/Wisc/OOD_Datasets/{out_dataset}", split='test',
                                    transform=large_transform, download=False)
            elif out_dataset == 'gaussian':
                testsetout = GaussianDataset(dataset_size =10000, img_size = 224,
                    transform=transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
            elif  out_dataset == 'celebA_ood':
                testsetout = celebAOodDataset()
            else:
                testsetout = tvs.datasets.ImageFolder(f"/content/drive/MyDrive/SP_OOD_Experiments/Wisc/OOD_Datasets/{out_dataset}",
                                            transform=large_transform)
            if out_dataset == 'celebA_ood':
                np.random.seed(0)
                subset = torch.utils.data.Subset(testsetout, np.random.choice(len(testsetout), 2000, replace=True))
                # testloaderOut =  
                num_samples = len(subset)
                val_size = int(0.2 * num_samples)  
                np.random.seed(0)
                val_indices = np.random.choice(num_samples, val_size, replace=False)
                np.random.seed(0)
                near_indices = np.random.choice(list(set(range(num_samples)) - set(val_indices)), num_samples - val_size, replace=False)

                subset_val = torch.utils.data.Subset(subset, val_indices)
                subset_near = torch.utils.data.Subset(subset, near_indices)
                sub_dataloader_dict = {}
                sub_dataloader_dict['sp_celebA'] = torch.utils.data.DataLoader(subset_near, batch_size=64,
                                                shuffle=False, num_workers=4) 
                dataloader_dict['ood']['near'].update(sub_dataloader_dict)

            else:
                np.random.seed(0)
                subset = torch.utils.data.Subset(testsetout, np.random.choice(len(testsetout), 2000, replace=False))
                testloaderOut = torch.utils.data.DataLoader(subset, batch_size=args.ood_batch_size,
                                                shuffle=False, num_workers=4)  

                num_samples = len(testsetout)
                val_size = int(0.2 * num_samples)  
                np.random.seed(0)
                val_indices = np.random.choice(num_samples, val_size, replace=False)
                np.random.seed(0)
                far_indices = np.random.choice(list(set(range(num_samples)) - set(val_indices)), num_samples - val_size, replace=False)

                subset_val = torch.utils.data.Subset(testsetout, val_indices)
                subset_far = torch.utils.data.Subset(testsetout, far_indices)
                sub_dataloader_dict = {}
                sub_dataloader_dict[dataset_name] = torch.utils.data.DataLoader(subset_far, batch_size=64, shuffle=False,
                                                                                num_workers=4)
                dataloader_dict['ood']['far'].update(sub_dataloader_dict)
                sub_dataloader_dict_val = {'sp_celebA': sub_dataloader_dict_val['sp_celebA']}
                sub_dataloader_dict_val[dataset_name] = torch.utils.data.DataLoader(subset_val, batch_size=64, shuffle=False,
                                                                                    num_workers=4)
                dataloader_dict['ood']['val'].update(sub_dataloader_dict_val)

        return dataloader_dict


    if 'imagenet' in id_name:
        if tvs_new:
            if isinstance(preprocessor,
                          tvs.transforms._presets.ImageClassification):
                mean, std = preprocessor.mean, preprocessor.std
            elif isinstance(preprocessor, tvs.transforms.Compose):
                temp = preprocessor.transforms[-1]
                mean, std = temp.mean, temp.std
            elif isinstance(preprocessor, BasePreprocessor):
                temp = preprocessor.transform.transforms[-1]
                mean, std = temp.mean, temp.std
            else:
                raise TypeError
        else:
            if isinstance(preprocessor, tvs.transforms.Compose):
                temp = preprocessor.transforms[-1]
                mean, std = temp.mean, temp.std
            elif isinstance(preprocessor, BasePreprocessor):
                temp = preprocessor.transform.transforms[-1]
                mean, std = temp.mean, temp.std
            else:
                raise TypeError
        imagenet_c_preprocessor = ImageNetCPreProcessor(mean, std)

    # weak augmentation for data_aux
    test_standard_preprocessor = get_default_preprocessor(id_name)

    dataloader_dict = {}
    data_info = DATA_INFO[id_name]

    # id
    sub_dataloader_dict = {}
    for split in data_info['id'].keys():
        dataset = ImglistDataset(
            name='_'.join((id_name, split)),
            imglist_pth=os.path.join(data_root,
                                     data_info['id'][split]['imglist_path']),
            data_dir=os.path.join(data_root,
                                  data_info['id'][split]['data_dir']),
            num_classes=data_info['num_classes'],
            preprocessor=preprocessor,
            data_aux_preprocessor=test_standard_preprocessor)
        dataloader = DataLoader(dataset, **loader_kwargs)
        sub_dataloader_dict[split] = dataloader
    dataloader_dict['id'] = sub_dataloader_dict

    # csid
    sub_dataloader_dict = {}
    for dataset_name in data_info['csid']['datasets']:
        dataset = ImglistDataset(
            name='_'.join((id_name, 'csid', dataset_name)),
            imglist_pth=os.path.join(
                data_root, data_info['csid'][dataset_name]['imglist_path']),
            data_dir=os.path.join(data_root,
                                  data_info['csid'][dataset_name]['data_dir']),
            num_classes=data_info['num_classes'],
            preprocessor=preprocessor
            if dataset_name != 'imagenet_c' else imagenet_c_preprocessor,
            data_aux_preprocessor=test_standard_preprocessor)
        dataloader = DataLoader(dataset, **loader_kwargs)
        sub_dataloader_dict[dataset_name] = dataloader
    dataloader_dict['csid'] = sub_dataloader_dict

    # ood
    dataloader_dict['ood'] = {}
    for split in data_info['ood'].keys():
        split_config = data_info['ood'][split]

        if split == 'val':
            # validation set
            dataset = ImglistDataset(
                name='_'.join((id_name, 'ood', split)),
                imglist_pth=os.path.join(data_root,
                                         split_config['imglist_path']),
                data_dir=os.path.join(data_root, split_config['data_dir']),
                num_classes=data_info['num_classes'],
                preprocessor=preprocessor,
                data_aux_preprocessor=test_standard_preprocessor)
            dataloader = DataLoader(dataset, **loader_kwargs)
            dataloader_dict['ood'][split] = dataloader
        else:
            # dataloaders for nearood, farood
            sub_dataloader_dict = {}
            for dataset_name in split_config['datasets']:
                dataset_config = split_config[dataset_name]
                dataset = ImglistDataset(
                    name='_'.join((id_name, 'ood', dataset_name)),
                    imglist_pth=os.path.join(data_root,
                                             dataset_config['imglist_path']),
                    data_dir=os.path.join(data_root,
                                          dataset_config['data_dir']),
                    num_classes=data_info['num_classes'],
                    preprocessor=preprocessor,
                    data_aux_preprocessor=test_standard_preprocessor)
                dataloader = DataLoader(dataset, **loader_kwargs)
                sub_dataloader_dict[dataset_name] = dataloader
            dataloader_dict['ood'][split] = sub_dataloader_dict

    return dataloader_dict
