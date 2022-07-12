from data.utils import DirLoader
from .experiments import experiments_loader
from .toy import toy_loader

data_loader = DirLoader(experiments_loader, toy_loader, directory='data')
