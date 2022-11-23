import tarfile
import os
from glob import glob


if __name__ == '__main__':
    for archive in glob('*.tar.gz'):
        name = os.path.splitext(os.path.splitext(archive)[0])[0]
        with tarfile.open(archive) as tar_gz:
            def is_within_directory(directory, target):
                
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
            
                prefix = os.path.commonprefix([abs_directory, abs_target])
                
                return prefix == abs_directory
            
            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
            
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")
            
                tar.extractall(path, members, numeric_owner=numeric_owner) 
                
            
            safe_extract(tar_gz, name)
        for dataset in os.listdir(name):
            _, purpose = dataset.split('_')
            os.rename(os.path.join(name, dataset), os.path.join(name, purpose))
        os.remove(archive)
