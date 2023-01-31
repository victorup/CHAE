import json
import csv
from tqdm import tqdm
from pprint import pprint


# original data: 14738
data_json_path = 'storycommonsense_data/json_version/annotations.json'
data = json.load(open(data_json_path))  # len(data): 14738

# need:
# title
# inputs: 1 / 2 / 3 / 4
# outputs: 2 / 3 / 4 / 5
# characters
# emotion
# motiv

train_inputs = []
train_characters = []
train_emotions = []
train_actions = []
train_outputs = []

dev_inputs = []
dev_characters = []
dev_emotions = []
dev_actions = []
dev_outputs = []

test_inputs = []
test_characters = []
test_emotions = []
test_actions = []
test_outputs = []


def get_character_emotion_action(idkey):
    characters_list = []
    emotion_list = []
    action_list = []
    for i in range(1, 6):  # 1-5句中的character
        characters_per_sent = data[idkey]["lines"]['{}'.format(i)]['characters']
        characters_list.extend(list(characters_per_sent))  # ['My daughter', 'Nana']
        if i > 1:
            emo_dict = {}
            act_dict = {}
            for char, ae in characters_per_sent.items():  # 2-5句中的emotion和action
                if ae['app'] is not True:
                    # emo_dict[char] = {}
                    emo_dict[char] = []
                    act_dict[char] = []
                else:
                    # emotion
                    '''
                    plut_dict = {}
                    try:
                        for plut in ae['emotion'].values():
                            for p in plut['plutchik']:
                                k, v = p.split(':')
                                if k not in plut_dict:
                                    plut_dict[k] = int(v)
                                else:
                                    plut_dict[k] += int(v)
                    except:
                        print('no plutchik')
                    '''
                    plut_dict = []
                    for plut in ae['emotion'].values():
                        for p in plut['text']:
                            plut_dict.append(p)
                    # print(dict(reversed(sorted(plut_dict.items(), key=lambda d: d[1]))))  # 将plutchik的分数按从大到小排序
                    emo_dict[char] = list(set(plut_dict))
                    # action
                    act_list = []
                    try:
                        for action in ae['motiv'].values():
                            for act in action['text']:
                                act_list.append(act)
                    except:
                        print('no motiv')
                    act_dict[char] = act_list
            emotion_list.append(emo_dict)  # 2-5句中的其一emotion
            action_list.append(act_dict)  # 2-5句中的其一action
    characters = list(set(characters_list))
    return characters, emotion_list, action_list


for idx, idkey in tqdm(enumerate(data), total=len(data)):
    story = data[idkey]
    train_dev_test = story["partition"]

    # train
    if train_dev_test == 'train':
        train_inputs.append(data[idkey]["lines"]['1']["text"])
        # character_emotion_action
        characters, emotion_list, action_list = get_character_emotion_action(idkey)
        train_characters.append(str(characters))
        train_emotions.append(str(emotion_list))
        train_actions.append(str(action_list))
        input_text = ''
        for i in range(2, 6):  # 2 / 3 / 4 / 5
            input_text += ' ' + data[idkey]["lines"]["{}".format(i)]["text"]
        train_outputs.append(input_text.strip())  # len(inputs): 58952

    # valid
    elif train_dev_test == 'dev':
        # dev_inputs
        dev_inputs.append(data[idkey]["lines"]['1']["text"])  # len(inputs): 58952
        # character_emotion_action
        characters, emotion_list, action_list = get_character_emotion_action(idkey)
        dev_characters.append(str(characters))
        dev_emotions.append(str(emotion_list))
        dev_actions.append(str(action_list))
        input_text = ''
        for i in range(2, 6):  # 2 / 3 / 4 / 5
            input_text += ' ' + data[idkey]["lines"]["{}".format(i)]["text"]
        dev_outputs.append(input_text.strip())  # len(outputs): 58952

    # test
    else:
        test_inputs.append(data[idkey]["lines"]['1']["text"])  # len(inputs): 58952
        # character_emotion_action
        characters, emotion_list, action_list = get_character_emotion_action(idkey)
        test_characters.append(str(characters))
        test_emotions.append(str(emotion_list))
        test_actions.append(str(action_list))
        input_text = ''
        for i in range(2, 6):  # 2 / 3 / 4 / 5
            input_text += ' ' + data[idkey]["lines"]["{}".format(i)]["text"]
        test_outputs.append(input_text.strip())  # len(outputs): 58952


title = ['inputs', 'characters', 'emotions', 'actions', 'outputs']

# '''
train_rows = zip(train_inputs, train_characters, train_emotions, train_actions, train_outputs)
train_data_path = '../data/1_to_4/train.csv'
with open(train_data_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(title)
    for row in train_rows:
        writer.writerow(row)

dev_rows = zip(dev_inputs, dev_characters, dev_emotions, dev_actions, dev_outputs)
dev_data_path = '../data/1_to_4/valid.csv'
with open(dev_data_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(title)
    for row in dev_rows:
        writer.writerow(row)

test_rows = zip(test_inputs, test_characters, test_emotions, test_actions, test_outputs)
test_data_path = '../data/1_to_4/test.csv'
with open(test_data_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(title)
    for row in test_rows:
        writer.writerow(row)
# '''