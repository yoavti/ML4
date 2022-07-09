import tarfile
import os
from glob import glob


if __name__ == '__main__':
    for archive in glob('*.tar.gz'):
        name = os.path.splitext(os.path.splitext(archive)[0])[0]
        with tarfile.open(archive) as tar_gz:
            tar_gz.extractall(name)
        for dataset in os.listdir(name):
            _, purpose = dataset.split('_')
            os.rename(os.path.join(name, dataset), os.path.join(name, purpose))
        os.remove(archive)
