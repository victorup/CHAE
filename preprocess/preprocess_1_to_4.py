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
train_outputs = []

dev_inputs = []
dev_outputs = []

test_inputs = []
test_outputs = []

for idkey in data:
    story = data[idkey]
    train_dev_test = story["partition"]
    if train_dev_test == 'train':
        train_inputs.append(data[idkey]["lines"]['1']["text"])
        input_text = ''
        for i in range(2, 6):  # 2 / 3 / 4 / 5
            input_text += ' ' + data[idkey]["lines"]["{}".format(i)]["text"]
        train_outputs.append(input_text.strip())  # len(inputs): 58952

    elif train_dev_test == 'dev':
        dev_inputs.append(data[idkey]["lines"]['1']["text"])  # len(inputs): 58952
        input_text = ''
        for i in range(2, 6):  # 2 / 3 / 4 / 5
            input_text += ' ' + data[idkey]["lines"]["{}".format(i)]["text"]
        dev_outputs.append(input_text.strip())  # len(outputs): 58952

    else:
        test_inputs.append(data[idkey]["lines"]['1']["text"])  # len(inputs): 58952
        input_text = ''
        for i in range(2, 6):  # 2 / 3 / 4 / 5
            input_text += ' ' + data[idkey]["lines"]["{}".format(i)]["text"]
        test_outputs.append(input_text.strip())  # len(outputs): 58952

title = ['inputs', 'outputs']
train_rows = zip(train_inputs, train_outputs)
train_data_path = '../data/1_to_4/train.csv'
with open(train_data_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(title)
    for row in train_rows:
        writer.writerow(row)

dev_rows = zip(dev_inputs, dev_outputs)
dev_data_path = '../data/1_to_4/valid.csv'
with open(dev_data_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(title)
    for row in dev_rows:
        writer.writerow(row)

test_rows = zip(test_inputs, test_outputs)
test_data_path = '../data/1_to_4/test.csv'
with open(test_data_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(title)
    for row in test_rows:
        writer.writerow(row)