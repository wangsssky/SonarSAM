# configuration for the models
import yaml


class Config_SAM:
    def __init__(self, config_path):

        with open(config_path, encoding='utf-8') as f:
            yaml_dict = yaml.load(f, Loader=yaml.FullLoader)

        # ----------- parse yaml ---------------#
        self.DATA_PATH = yaml_dict['DATA_PATH']
        self.IMAGE_LIST_PATH = yaml_dict['IMAGE_LIST_PATH']

        self.RANDOM_SEED = yaml_dict['RANDOM_SEED']

        self.MODEL_NAME = yaml_dict['MODEL_NAME']
        self.MODEL_DIR = yaml_dict['MODEL_DIR']
        self.SAM_NAME = yaml_dict['SAM_NAME']
        self.SAM_CHECKPOINT = yaml_dict['SAM_CHECKPOINT']
        self.IS_FINETUNE_IMAGE_ENCODER = yaml_dict['IS_FINETUNE_IMAGE_ENCODER']
        self.USE_ADAPTATION = yaml_dict['USE_ADAPTATION']
        self.ADAPTATION_TYPE = yaml_dict['ADAPTATION_TYPE']
        self.HEAD_TYPE = yaml_dict['HEAD_TYPE']

        self.EPOCH_NUM = yaml_dict['EPOCH_NUM']
        self.RESUME_FROM = yaml_dict['RESUME_FROM']

        self.TRAIN_BATCHSIZE = yaml_dict['TRAIN_BATCHSIZE']
        self.VAL_BATCHSIZE = yaml_dict['VAL_BATCHSIZE']

        self.OPTIMIZER = yaml_dict['OPTIMIZER']
        self.WEIGHT_DECAY = yaml_dict['WEIGHT_DECAY']
        self.MOMENTUM = yaml_dict['MOMENTUM']
        self.LEARNING_RATE = float(yaml_dict['LEARNING_RATE'])
        self.WARM_LEN = yaml_dict['WARM_LEN']

        self.INPUT_SIZE = yaml_dict['INPUT_SIZE']
        self.OUTPUT_CHN = yaml_dict['OUTPUT_CHN']
        
        self.PRT_LOSS = yaml_dict['PRT_LOSS']
        self.VISUALIZE = yaml_dict['VISUALIZE']
        self.EVAL_METRIC = yaml_dict['EVAL_METRIC']


