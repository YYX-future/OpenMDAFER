import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
import bisect

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

train_transform = transforms.Compose([transforms.Resize([224, 224]),
                                      transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                                      transforms.RandomGrayscale(p=0.2),
                                      transforms.RandomHorizontalFlip(p=0.5),
                                      transforms.RandomAffine(degrees=(-15, 15), translate=(0.05, 0.05),
                                                              scale=(0.9, 1.05), fill=0),
                                      transforms.ToTensor(),
                                      normalize])  # transform [0,255] to [0,1]

test_transform = transforms.Compose(
    [transforms.Resize([224, 224]), transforms.ToTensor(), normalize])  # transform [0,255] to [0,1]


class TrainDataset(Dataset):
    def __init__(self, image_txt_paths, open_class, transform=None, ensemble_source=False):
        self.data = []
        self.labels = []
        self.transform = transform

        if ensemble_source:
            for image_txt_path in image_txt_paths:
                with open(image_txt_path, 'r') as fh_image:
                    for line in fh_image.readlines():
                        line = line.strip().split()
                        label = int(line[-1])
                        file = line[0].strip()
                        if label != open_class:
                            self.data.append((file, label))
                            self.labels.append(label)

        else:
            with open(image_txt_paths, 'r') as fh_image:
                for line in fh_image.readlines():
                    line = line.strip().split()
                    label = int(line[-1])
                    file = line[0].strip()
                    if label != open_class:
                        self.data.append((file, label))
                        self.labels.append(label)

    def __getitem__(self, index):
        fn, label = self.data[index]
        img = Image.open(fn).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.data)


class TestDataset(Dataset):
    def __init__(self, image_txt_path, open_class, transform=None, is_tf=False):

        fh_image = open(image_txt_path, 'r')
        self.data = []
        self.labels = []
        self.transform = transform
        if is_tf:
            for line in fh_image.readlines():
                line = line.strip()
                line = line.split()
                label = int(line[-1])
                file = line[0].strip()
                if label == open_class:
                    self.data.append((file, label))
                    self.labels.append(label)

        else:
            for line in fh_image.readlines():
                line = line.strip()
                line = line.split()
                label = int(line[-1])
                file = line[0].strip()
                self.data.append((file, label))
                self.labels.append(label)

    def __getitem__(self, index):  # 返回tensor，标签
        fn, label = self.data[index]
        img = Image.open(fn).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.data)


class ConcatDataset(Dataset):
    def __init__(self, train_datasets):
        self.datasets = list(train_datasets)
        self.cumulative_sizes = self.cumsum([len(dataset) for dataset in self.datasets])

    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            l = e
            r.append(l + s)
            s += l
        return r

    def isMulti(self):
        return True

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx], dataset_idx


def load_training_tgt(root_path, source_paths, batch_size, open_class, transform_type=None, ensemble_source=False):

    if ensemble_source:
        source_paths = [os.path.join(root_path, path) for path in source_paths if path is not None]
    else:
        source_paths = os.path.join(root_path, source_paths)

    train_data = TrainDataset(source_paths, open_class, transform_type, ensemble_source)

    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, drop_last=True,
                              pin_memory=True, num_workers=4)  # sampler和shuffle不能同时为真

    return train_loader


def load_testing(root_path, txt_path, batch_size, open_class, is_tf=False):

    test_data = TestDataset(os.path.join(root_path, txt_path), open_class, test_transform, is_tf)

    return test_data


def load_training_src(root_path, source_paths, open_class, transform_type=None, ensemble_source=False):

    if ensemble_source:
        source_paths = [os.path.join(root_path, path) for path in source_paths if path is not None]
    else:
        source_paths = os.path.join(root_path, source_paths)

    train_data = TrainDataset(source_paths, open_class, transform_type, ensemble_source)

    return train_data


def get_loaders_epoch(root, args, open_class, ensemble=False):

    # domains = ['RAF', 'Oulu', 'FER2013', 'Aff',]

    sources = args.src_domain.copy()
    sources.remove(args.tgt_domain)

    tgt_domain = args.tgt_domain
    src_paths = [source + '.txt' for source in sources]
    datasets = []

    for i in range(len(src_paths)):
        datasets.append(load_training_src(root, src_paths[i], open_class, train_transform))
    if ensemble:
        datasets.append(load_training_src(root, src_paths, open_class, train_transform, ensemble_source=True))

    dataset = ConcatDataset(datasets)
    src_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4,
                            pin_memory=True, drop_last=True)

    if tgt_domain in ["JAFFE", "CK", "Oulu"]:
        tgt_train = f"{tgt_domain}.txt"
        tgt_train_dl = load_training_tgt(root, tgt_train, args.batch_size, open_class, train_transform)
        tgt_test_dataset = load_testing(root, tgt_train, args.batch_size, open_class)
        tgt_test_dl = DataLoader(tgt_test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4,
                                 pin_memory=True, drop_last=False)
        return src_loader, tgt_train_dl, tgt_test_dl

    else:
        test_datasets = []
        tgt_train = f"{tgt_domain}_train.txt"
        tgt_test = f"{tgt_domain}_test.txt"
        tgt_train_dl = load_training_tgt(root, tgt_train, args.batch_size, open_class, train_transform)

        test_datasets.append(load_testing(root, tgt_train, args.batch_size, open_class, is_tf=True))
        test_datasets.append(load_testing(root, tgt_test, args.batch_size, open_class))
        test_datasets = ConcatDataset(test_datasets)
        tgt_test_dl = DataLoader(test_datasets, batch_size=args.batch_size, shuffle=False, num_workers=4,
                                 pin_memory=True, drop_last=False)

        return src_loader, tgt_train_dl, tgt_test_dl
