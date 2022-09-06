import os

from engine.tools.utils import listdir_nohidden
from engine.datasets.imagenet import read_classnames
from engine.datasets.benchmark import Benchmark

TO_BE_IGNORED = ["README.txt"]


class ImageNetA(Benchmark):
    """ImageNet-A(dversarial).

    This dataset is used for testing only.
    """

    dataset_name = "imagenet-adversarial"

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATA_DIR))
        self.dataset_dir = os.path.join(root, self.dataset_name)
        self.image_dir = os.path.join(self.dataset_dir, "imagenet-a")

        text_file = os.path.join(self.dataset_dir, "classnames.txt")
        classnames = read_classnames(text_file)

        data = self.read_data(classnames)

        super().__init__(train=data, val=data, test=data)

    def read_data(self, classnames):
        image_dir = self.image_dir
        folders = listdir_nohidden(image_dir, sort=True)
        folders = [f for f in folders if f not in TO_BE_IGNORED]
        items = []

        for label, folder in enumerate(folders):
            imnames = listdir_nohidden(os.path.join(image_dir, folder))
            classname = classnames[folder]
            for imname in imnames:
                impath = os.path.join(image_dir, folder, imname)
                item = {"impath": impath, "label": label, "classname": classname}
                items.append(item)

        return items