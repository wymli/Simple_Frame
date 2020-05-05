from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau

# from datasets import *
from models.graph_classifiers.DGCNN import DGCNN
from models.graph_classifiers.DeepMultisets import DeepMultisets
from models.graph_classifiers.MolecularFingerprint import MolecularFingerprint
# from models.schedulers.ECCScheduler import ECCLR
# from models.utils.EarlyStopper import Patience, GLStopper
from models.graph_classifiers.GIN import GIN
from models.graph_classifiers.DiffPool import DiffPool
from models.graph_classifiers.ECC import ECC
from models.graph_classifiers.GraphSAGE import GraphSAGE

from models.losses import *
# from models.modules import (BinaryClassificationLoss, MulticlassClassificationLoss,
# NN4GMulticlassClassificationLoss, DiffPoolMulticlassClassificationLoss)

from copy import deepcopy
# from .utils import read_config_file


def getConfig(key: str, value: str):
    dict = {
        # "datasets": {
        #     'NCI1': NCI1,
        #     'IMDB-BINARY': IMDBBinary,
        #     'IMDB-MULTI': IMDBMulti,
        #     'COLLAB': Collab,
        #     'REDDIT-BINARY': RedditBinary,
        #     'REDDIT-MULTI-5K': Reddit5K,
        #     'PROTEINS': Proteins,
        #     'ENZYMES': Enzymes,
        #     'DD': DD,
        #     "BBBP":BBBP,
        # },
        "models": {
            'GIN': GIN,
            'ECC': ECC,
            "DiffPool": DiffPool,
            "DGCNN": DGCNN,
            "MolecularFingerprint": MolecularFingerprint,
            "DeepMultisets": DeepMultisets,
            "GraphSAGE": GraphSAGE
        },

        "losses": {
            #分类任务 损失
            'BinaryClassificationLoss': BinaryClassificationLoss,
            'MulticlassClassificationLoss': MulticlassClassificationLoss,
            'NN4GMulticlassClassificationLoss': NN4GMulticlassClassificationLoss,
            'DiffPoolMulticlassClassificationLoss': DiffPoolMulticlassClassificationLoss,
        },

        "optimizers": {
            'Adam': Adam,
            'SGD': SGD
        },

        "schedulers": {
            'StepLR': StepLR,
            # 'ECCLR': ECCLR,
            'ReduceLROnPlateau': ReduceLROnPlateau
        },

        # "early_stoppers": {
        #     'GLStopper': GLStopper,
        #     'Patience': Patience
        # },

    }
    return dict[key][value]
