import os

from torch.utils.data import Dataset, DataLoader
import torchvision.datasets as Datasets
import torchvision.transforms as transforms
import pandas as pd
from PIL import Image

from sklearn.model_selection import train_test_split
import Helpers as hf

# Function to select the dataset to train and test
def train_test_setting(dataset, perc_black, seed, size_test, args):
    # label_col: optional, only read by callers that need supervised labels alongside
    # the (image, id) pairs (e.g. the downstream classifier probe). None (default) for
    # every existing caller -- unsupervised training/eval never touches this.
    label_col = getattr(args, 'label_col', None)
    # Dataset
    if args.dataset == 'fitz':
        print("Fitzpatrick dataset!")
        experiment_data = train_test_fitz(perc_black, seed, size_test, args, label_col)
    elif args.dataset == 'ai4skin':
        print("AI4SkIN dataset!")
        experiment_data = train_test_ai4skin(perc_black, seed, size_test, args, label_col)
    else:
        raise ValueError(f"Dataset not found: {args.dataset!r}. Please select 'fitz' or 'ai4skin'.")
    return experiment_data

# Function to split in train and test the fitzpatrick17k dataset
def train_test_fitz(perc_black, seed, size_test, args, label_col=None):
    # dataset CSV with fitzpatrick scale labels and imagefile names
    df = pd.read_csv(args.dataset_root + "../fitzpatrick17k.csv")

    # ligh skin
    white_df = df[df['fitzpatrick_scale'].isin([1, 2])]
    # dark skin
    black_df = df[df['fitzpatrick_scale'].isin([5, 6])]

    # Extract a random subsample of light skin samples for test
    print('Light df has length: ', len(white_df))
    test_w = white_df.sample(n=size_test, random_state=seed)
    white_df = white_df.drop(test_w.index)
    print('Light test df has length: ', len(test_w), 'the dataframe is now: ', len(white_df))

    # Extract a random subsample of dark skin samples for test
    print('Dark df has length: ', len(black_df))
    test_b = black_df.sample(n=size_test, random_state=seed)
    black_df = black_df.drop(test_b.index)
    print('Dark test df has length: ', len(test_b), 'the dataframe is now: ', len(black_df))

    # Fixed-size training set, resampled WITH replacement (paper protocol)
    n_black = round(perc_black * args.train_size)
    n_white = args.train_size - n_black
    white_df = white_df.sample(n=n_white, replace=True, random_state=seed)
    black_df = black_df.sample(n=n_black, replace=True, random_state=seed)
    train_df = pd.concat([white_df, black_df])

    # Class train/test distribution (according to fitzpatrick scale)
    print(train_df["fitzpatrick_scale"].value_counts())
    print(test_w["fitzpatrick_scale"].value_counts())
    print(test_b["fitzpatrick_scale"].value_counts())

    # Convert to fitz_dataset class
    train_dataset = fitz_dataset(train_df, args.dataset_root, args.image_size, test=False, label_col=label_col)
    test_dataset_w = fitz_dataset(test_w, args.dataset_root, args.image_size, test=True, label_col=label_col)
    test_dataset_b = fitz_dataset(test_b, args.dataset_root, args.image_size, test=True, label_col=label_col)

    return train_dataset, test_dataset_w, test_dataset_b, 'light_skin', 'dark_skin'

# Dataset class for fitzpatrick17k dataset
class fitz_dataset(Dataset):
    def __init__(self, dataframe, img_dir, image_size, test=False, label_col=None):
        self.dataframe = dataframe
        imgs = os.listdir(img_dir)
        imgs_hash = set(x.split('.')[0] for x in imgs)
        # Preserve row order/duplicates: with-replacement resampling relies on
        # the same hash appearing multiple times surviving through to __getitem__.
        mask = dataframe['md5hash'].isin(imgs_hash)
        self.ids = dataframe.loc[mask, 'md5hash'].tolist()
        self.labels = dataframe.loc[mask, label_col].tolist() if label_col else None
        self.images_fps = [img_dir + h + '.jpg' for h in self.ids]
        if test:
            self.transform = transforms.Compose([transforms.Resize(image_size),
                                transforms.CenterCrop(image_size)
                                ])
        else:
            self.transform = transforms.Compose([transforms.Resize(image_size),
                                transforms.CenterCrop(image_size),
                                transforms.RandomHorizontalFlip(0.5)
                                ])


    def __len__(self):
        return len(self.images_fps)

    def __getitem__(self, idx):
        # Get features and target
        # read data
        img = Image.open(self.images_fps[idx])
        img = img.convert("RGB")

        image = self.transform(img)

        return hf.image_to_numpy(image), self.ids[idx]

# Function to split in train and test the AI4SkIN histopathology dataset (WSI patches)
def train_test_ai4skin(perc_hcuv, seed, size_test, args, label_col=None):
    # Precomputed patch metadata: columns patch_path, wsi, center, gt, subset, x, y, tissue_frac
    df = pd.read_csv(args.ai4skin_patches_csv, header=None,
                      names=["patch_path", "wsi", "center", "gt", "subset", "x", "y", "tissue_frac"],
                      low_memory=False)

    train_pool = df[df['subset'] == 'train']
    test_pool = df[df['subset'] == 'test']

    # Fixed held-out test set, drawn without replacement from the pre-split test patches
    husc_test_pool = test_pool[test_pool['center'] == 'HUSC']
    hcuv_test_pool = test_pool[test_pool['center'] == 'HCUV']
    print('HUSC test pool has length: ', len(husc_test_pool))
    print('HCUV test pool has length: ', len(hcuv_test_pool))
    test_husc = husc_test_pool.sample(n=size_test, random_state=seed)
    test_hcuv = hcuv_test_pool.sample(n=size_test, random_state=seed)

    # Fixed-size training set, resampled WITH replacement (paper protocol)
    husc_train_pool = train_pool[train_pool['center'] == 'HUSC']
    hcuv_train_pool = train_pool[train_pool['center'] == 'HCUV']
    n_hcuv = round(perc_hcuv * args.train_size)
    n_husc = args.train_size - n_hcuv
    husc_train = husc_train_pool.sample(n=n_husc, replace=True, random_state=seed)
    hcuv_train = hcuv_train_pool.sample(n=n_hcuv, replace=True, random_state=seed)
    train_df = pd.concat([husc_train, hcuv_train])

    # Class train/test distribution (according to imaging center)
    print(train_df["center"].value_counts())
    print(test_husc["center"].value_counts())
    print(test_hcuv["center"].value_counts())

    # Convert to ai4skin_dataset class
    train_dataset = ai4skin_dataset(train_df, args.ai4skin_patches_root, args.image_size, test=False, label_col=label_col)
    test_dataset_husc = ai4skin_dataset(test_husc, args.ai4skin_patches_root, args.image_size, test=True, label_col=label_col)
    test_dataset_hcuv = ai4skin_dataset(test_hcuv, args.ai4skin_patches_root, args.image_size, test=True, label_col=label_col)

    return train_dataset, test_dataset_husc, test_dataset_hcuv, 'HUSC', 'HCUV'

# Dataset class for the AI4SkIN histopathology dataset (WSI patches)
class ai4skin_dataset(Dataset):
    def __init__(self, dataframe, patches_root, image_size, test=False, label_col=None):
        self.dataframe = dataframe
        # Patches come from a precomputed extraction manifest, so (unlike fitz_dataset)
        # there's no need to check file existence against the filesystem.
        self.ids = dataframe['patch_path'].tolist()
        self.labels = dataframe[label_col].tolist() if label_col else None
        self.images_fps = [os.path.join(patches_root, p) for p in self.ids]
        if test:
            self.transform = transforms.Compose([transforms.Resize(image_size),
                                transforms.CenterCrop(image_size)
                                ])
        else:
            self.transform = transforms.Compose([transforms.Resize(image_size),
                                transforms.CenterCrop(image_size),
                                transforms.RandomHorizontalFlip(0.5)
                                ])

    def __len__(self):
        return len(self.images_fps)

    def __getitem__(self, idx):
        img = Image.open(self.images_fps[idx])
        img = img.convert("RGB")

        image = self.transform(img)

        return hf.image_to_numpy(image), self.ids[idx]

# Function to load an external dataset (i.e., PASSION and ISIC)
def load_external_test(args):
    root_data = args.test_set_root
    if 'passion' in root_data:
        print("PASSION dataset")
        df = pd.read_csv(root_data + "label.csv")
        external_test = passion_dataset(df, root_data + "images_resized/", args.image_size)
    elif 'ISIC' in root_data:
        print("ISIC dataset")
        external_test = ISIC_dataset(root_data + "images_resized/", args.image_size)
    else:
        raise("Dataset not found!")
    return external_test

# Dataset class for ISIC dataset
class ISIC_dataset(Dataset):
    def __init__(self, img_dir, image_size, test=True):
        self.imgs = os.listdir(img_dir)
        self.images_fps = [img_dir + x for x in self.imgs]
        print("Number of images: ", len(self.images_fps))
        if test:
            self.transform = transforms.Compose([transforms.Resize(image_size),
                                transforms.CenterCrop(image_size)
                                ])
        else:
            self.transform = transforms.Compose([transforms.Resize(image_size),
                                transforms.CenterCrop(image_size),
                                transforms.RandomHorizontalFlip(0.5)
                                ])


    def __len__(self):
        return len(self.images_fps)

    def __getitem__(self, idx):
        # Get features and target
        # read data
        img = Image.open(self.images_fps[idx])
        img = img.convert("RGB")

        image = self.transform(img)

        return hf.image_to_numpy(image), self.imgs[idx]

# Dataset class for PASSION dataset
class passion_dataset(Dataset):
    def __init__(self, dataframe, img_dir, image_size, test=True):
        self.dataframe = dataframe
        self.imgs = os.listdir(img_dir)
        self.images_fps = [img_dir + x for x in self.imgs]
        print("Number of images: ", len(self.images_fps))
        if test:
            self.transform = transforms.Compose([transforms.Resize(image_size),
                                transforms.CenterCrop(image_size)
                                ])
        else:
            self.transform = transforms.Compose([transforms.Resize(image_size),
                                transforms.CenterCrop(image_size),
                                transforms.RandomHorizontalFlip(0.5)
                                ])
    def __len__(self):
        return len(self.images_fps)

    def __getitem__(self, idx):
        # Get features and target
        # read data
        img = Image.open(self.images_fps[idx])
        img = img.convert("RGB")

        image = self.transform(img)

        return hf.image_to_numpy(image), self.imgs[idx]
