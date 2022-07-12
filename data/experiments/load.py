from data.utils import DirLoader
from .ARFF import arff_loader
from .bioconductor import bioconductor_loader
from .Datamicroarray import datamicroarray_loader
from .scikit_feature_datasets import scikit_feature_loader

experiments_loader = DirLoader(arff_loader, bioconductor_loader, datamicroarray_loader, scikit_feature_loader,
                               directory='experiments')
