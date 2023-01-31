import json
import csv

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
    emotion_list = []
    action_list = []
    emo_dict = {}
    act_dict = {}
    characters_per_sent = data[idkey]["lines"]['{}'.format(i)]['characters']
    char_score_dict = {}
    for char, ae in characters_per_sent.items():
        if ae['app'] is not True:
            emo_dict[char] = []
            act_dict[char] = []
            char_score_dict[char] = 0
        else:
            # emotion
            try:
                plut_dict = {}
                if len(ae['emotion'].values()) == 0:
                    emo_dict[char] = []
                    char_score_dict[char] = 0
                else:
                    for plut in ae['emotion'].values():
                        if len(plut['plutchik']) == 0:
                            continue
                        for p in plut['plutchik']:
                            p_label, p_score = p.split(':')
                            if p_label not in plut_dict:
                                plut_dict[p_label] = int(p_score)
                            else:
                                plut_dict[p_label] += int(p_score)
                    # 对plut_dict根据value大小进行排序
                    if plut_dict == {}:
                        raise KeyError
                    sorted_plut_dict = sorted(plut_dict.items(), key=lambda e: e[1], reverse=True)  # 按字典值降序排序
                    plut_list = [sorted_plut_dict[0][0]]  # 取最大的key
                    char_score_dict[char] = sorted_plut_dict[0][1]
                    # for k, v in sorted_plut_dict.items():
                    #     if v >= 4:
                    #         plut_list.append(k)
                    emo_dict[char] = plut_list
            except KeyError:
                # print('no plut')
                emo_dict[char] = []
                char_score_dict[char] = 0
            # action
            try:
                act_list = []
                for action in ae['motiv'].values():
                    for act in action['text']:
                        act_list.append(act)
                act_dict[char] = act_list
            except:
                # print('no motiv')
                act_dict[char] = []

    characters = list(dict(sorted(char_score_dict.items(), key=lambda e:e[1], reverse=True)))[:2]

    new_emo_dict = {}
    new_act_dict = {}
    for char in characters:
        new_emo_dict[char] = emo_dict[char]
        new_act_dict[char] = act_dict[char]
    emotion_list.append(new_emo_dict)
    action_list.append(new_act_dict)
    return characters, emotion_list, action_list


for idkey in data:
    story = data[idkey]
    train_dev_test = story["partition"]
    if train_dev_test == 'train':
        for i in range(1, 5):  # 1 / 2 / 3 / 4
            input_text = ''
            for j in range(1, i + 1):
                input_text += ' ' + data[idkey]["lines"]["{}".format(j)]["text"]
            train_inputs.append(input_text.strip())  # len(inputs): 58952
        for i in range(2, 6):  # 2 / 3 / 4 / 5
            train_outputs.append(data[idkey]["lines"]["{}".format(i)]["text"])  # len(outputs): 58952
            characters, emotion_list, action_list = get_character_emotion_action(idkey)
            train_characters.append(str(characters))
            train_emotions.append(str(emotion_list))
            train_actions.append(str(action_list))

    elif train_dev_test == 'dev':
        for i in range(1, 5):  # 1 / 2 / 3 / 4
            input_text = ''
            for j in range(1, i + 1):
                input_text += ' ' + data[idkey]["lines"]["{}".format(j)]["text"]
            dev_inputs.append(input_text.strip())  # len(inputs): 58952
        for i in range(2, 6):  # 2 / 3 / 4 / 5
            dev_outputs.append(data[idkey]["lines"]["{}".format(i)]["text"])  # len(outputs): 58952
            characters, emotion_list, action_list = get_character_emotion_action(idkey)
            dev_characters.append(str(characters))
            dev_emotions.append(str(emotion_list))
            dev_actions.append(str(action_list))

    else:
        for i in range(1, 5):  # 1 / 2 / 3 / 4
            input_text = ''
            for j in range(1, i + 1):
                input_text += ' ' + data[idkey]["lines"]["{}".format(j)]["text"]
            test_inputs.append(input_text.strip())  # len(inputs): 58952
        for i in range(2, 6):  # 2 / 3 / 4 / 5
            test_outputs.append(data[idkey]["lines"]["{}".format(i)]["text"])  # len(outputs): 58952
            characters, emotion_list, action_list = get_character_emotion_action(idkey)
            test_characters.append(str(characters))
            test_emotions.append(str(emotion_list))
            test_actions.append(str(action_list))

# '''
title = ['inputs', 'characters', 'emotions', 'actions', 'outputs']

train_rows = zip(train_inputs, train_characters, train_emotions, train_actions, train_outputs)
train_data_path = '../data/n_to_1_new/train.csv'
with open(train_data_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(title)
    for row in train_rows:
        writer.writerow(row)

dev_rows = zip(dev_inputs, dev_characters, dev_emotions, dev_actions, dev_outputs)
dev_data_path = '../data/n_to_1_new/valid.csv'
with open(dev_data_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(title)
    for row in dev_rows:
        writer.writerow(row)

test_rows = zip(test_inputs, test_characters, test_emotions, test_actions, test_outputs)
test_data_path = '../data/n_to_1_new/test.csv'
with open(test_data_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(title)
    for row in test_rows:
        writer.writerow(row)
# '''