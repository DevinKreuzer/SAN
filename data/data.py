"""
    File to load dataset based on user control from main file
"""
from data.molecules import MoleculeDataset
from data.SBMs import SBMsDataset
from data.molhiv import MolHIVDataset
from data.molpcba import MolPCBADataset

def LoadData(DATASET_NAME):
    """
        This function is called in the main_xx.py file 
        returns:
        ; dataset object
    """

    # handling for (ZINC) molecule dataset
    if DATASET_NAME == 'ZINC':
        return MoleculeDataset(DATASET_NAME)

    # handling for SBM datasets
    SBM_DATASETS = ['SBM_CLUSTER', 'SBM_PATTERN']
    if DATASET_NAME in SBM_DATASETS: 
        return SBMsDataset(DATASET_NAME)
    
    if DATASET_NAME == 'MOL-HIV':
        return MolHIVDataset(DATASET_NAME)
    
    if DATASET_NAME == 'MOL-PCBA':
        return MolPCBADataset(DATASET_NAME)
    
