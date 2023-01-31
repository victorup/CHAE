from torch.utils.data import Dataset
import csv
import numpy as np


class MyDataset(Dataset):
    def __init__(self, data_path):
        self.inputs = []
        self.outputs = []
        # self.sentence_x_len = []
        # self.context_x = []
        # self.context_x_len = []
        # self.char_idx_next = []
        # self.plutchik_x = []
        # self.maslow_x = []
        # self.reiss_x = []
        # self.sentence_y = []
        # self.sentence_y_len = []
        with open(data_path, "r", encoding="utf8") as csvfile:
            self.data = csv.reader(csvfile)
            for idx, line in enumerate(self.data):
                # if idx > 500:
                #     break
                self.inputs.append(line[0])
                self.outputs.append(line[1])
                # self.sentence_x_len.append(eval(line[1]))
                # self.context_x.append(eval(line[2]))
                # self.context_x_len.append(eval(line[3]))
                # self.char_idx_next.append(eval(line[4]))
                # self.plutchik_x.append(eval(line[5]))
                # self.maslow_x.append(eval(line[6]))
                # self.reiss_x.append(eval(line[7]))
                # self.sentence_y.append(eval(line[8]))
                # self.sentence_y_len.append(eval(line[9]))

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.outputs[idx] \
               # self.sentence_x_len[idx], \ self.context_x[idx], self.context_x_len[idx], \
               # self.char_idx_next[idx], self.plutchik_x[idx], self.maslow_x[idx], self.reiss_x[idx], \
               # self.sentence_y[idx], self.sentence_y_len[idx]


def get_pmr_np(config, pmr_all, word_dict):
    if config.gpt2:
        pmr_np = np.zeros((len(pmr_all), 1)).astype('int32')
        tokenizer = word_dict
        for idx, pmr in enumerate(pmr_all):
            value = tokenizer.encode(pmr)[:1]
            pmr_np[idx, :] = np.array(value)
    else:
        pmr_np = np.zeros((len(pmr_all), 1)).astype('int32')
        for idx, pmr in enumerate(pmr_all):
            if pmr in word_dict:
                value = word_dict[pmr]
            else:
                value = word_dict['unk']
            pmr_np[idx, :] = np.array(value)
    return pmr_np


def get_pmr(config, word_dict):
    plutchik = ["joy", "trust", "fear", "surprise", "sadness", "disgust", "anger", "anticipation"]
    maslow = ['physiological', 'love', 'spiritual', 'esteem', 'stability']
    reiss = ['status', 'approval', 'tranquility', 'competition', 'health', 'family',
             'romance', 'food', 'indep', 'power', 'order', 'curiosity', 'serenity',
             'honor', 'belonging', 'contact', 'savings', 'idealism', 'rest']  # 19个reiss
    p_np = get_pmr_np(config, plutchik, word_dict)
    m_np = get_pmr_np(config, maslow, word_dict)
    r_np = get_pmr_np(config, reiss, word_dict)

    return p_np, m_np, r_np