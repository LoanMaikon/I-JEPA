from torchvision.io import read_image, ImageReadMode
from torch.utils.data import Dataset
from glob import glob
import scipy.io
import os

class ImageNetDataset(Dataset):
    def __init__(self, operation, dataset_folder_path, transform):
        self.operation = operation
        self.dataset_folder_path = dataset_folder_path
        self.transform = transform

        self.images = []
        self.labels = []

        validation_gd_path = f"{dataset_folder_path}/ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt"
        meta_path = f"{dataset_folder_path}/ILSVRC2012_devkit_t12/data/meta.mat"

        data = scipy.io.loadmat(open(meta_path, "rb"))
        synsets = data.get('synsets')

        id_to_wnid = {}
        wnid_to_class = {}
        for synset in synsets:
            if int(synset['num_children'][0][0][0]) == 0:
                wnid = synset['WNID'][0][0]
                words = synset['words'][0][0]

                wnid_to_class[wnid] = words
                id_to_wnid[int(synset['ILSVRC2012_ID'][0][0][0])] = wnid

        classes = sorted(wnid_to_class.values())
        class_to_id = {cls: idx for idx, cls in enumerate(classes)}

        train_images_per_class = {}
        train_wnid = os.listdir(self.dataset_folder_path + "train/")
        for wnid in train_wnid:
            _images = glob(self.dataset_folder_path + "train/" + wnid + "/*.JPEG")
            train_images_per_class[wnid] = _images

        match self.operation:
            case "train":
                for wnid, _images in train_images_per_class.items():
                    self.images.extend(_images)
                    class_name = wnid_to_class[wnid]
                    class_id = class_to_id[class_name]
                    self.labels.extend([class_id] * len(_images))

            case "test": # Using validation set as test set
                val_images = sorted(glob(self.dataset_folder_path + "val/*.JPEG"))

                idx_to_wnid = {}
                with open(validation_gd_path, "r") as file:
                    for idx, line in enumerate(file):
                        line = line.strip()
                        if not line:
                            continue
                        wnid = id_to_wnid[int(line)]
                        idx_to_wnid[idx] = wnid
                
                for idx, image in enumerate(val_images):
                    wnid = idx_to_wnid[idx]

                    self.images.append(image)
                    class_name = wnid_to_class[wnid]
                    class_id = class_to_id[class_name]
                    self.labels.append(class_id)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = read_image(self.images[idx], ImageReadMode.RGB)

        return self.transform(image), self.labels[idx]
