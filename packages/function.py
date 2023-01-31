import logging
from transformers.models.bart.tokenization_bart import BartTokenizer
from configs import Config


def collate_fn(batch):
    """
    计算该batch中的所有sample的最长的input，并且将其他input的长度向其对齐
    :param batch: batch中是MyDataset中__getitem__返回的元素，形式为tuple。for us是tuple len为6的输入源，
    :return:
    """
    inputs = [b[0] for b in batch]
    outputs = [b[1] for b in batch]

    # inputs_tensor = tokenizer(inputs, max_length=1024, return_tensors='pt', padding='longest')['input_ids']
    inputs_tensor = tokenizer(inputs, max_length=1024, return_tensors='pt', padding='longest')
    outputs_tensor = tokenizer(outputs, max_length=1024, return_tensors='pt', padding='longest')['input_ids']
    inputs_tensor['labels'] = outputs_tensor
    # return inputs_tensor, outputs_tensor
    return inputs_tensor


def create_logger(log_path):
    """
    将日志输出到日志文件和控制台
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # 创建一个handler，用于写入日志文件
    file_handler = logging.FileHandler(filename=log_path)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    # 创建一个handler，用于将日志输出到控制台
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    console.setFormatter(formatter)
    logger.addHandler(console)

    return logger


def get_config():
    config = Config()
    return config


config = get_config()
tokenizer = BartTokenizer.from_pretrained(config.pretrained_name)