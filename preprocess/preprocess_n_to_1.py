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
        for i in range(1, 5):  # 1 / 2 / 3 / 4
            input_text = ''
            for j in range(1, i + 1):
                input_text += ' ' + data[idkey]["lines"]["{}".format(j)]["text"]
            train_inputs.append(input_text.strip())  # len(inputs): 58952
        for i in range(2, 6):  # 2 / 3 / 4 / 5
            train_outputs.append(data[idkey]["lines"]["{}".format(i)]["text"])  # len(outputs): 58952

    elif train_dev_test == 'dev':
        for i in range(1, 5):  # 1 / 2 / 3 / 4
            input_text = ''
            for j in range(1, i + 1):
                input_text += ' ' + data[idkey]["lines"]["{}".format(j)]["text"]
            dev_inputs.append(input_text.strip())  # len(inputs): 58952
        for i in range(2, 6):  # 2 / 3 / 4 / 5
            dev_outputs.append(data[idkey]["lines"]["{}".format(i)]["text"])  # len(outputs): 58952

    else:
        for i in range(1, 5):  # 1 / 2 / 3 / 4
            input_text = ''
            for j in range(1, i + 1):
                input_text += ' ' + data[idkey]["lines"]["{}".format(j)]["text"]
            test_inputs.append(input_text.strip())  # len(inputs): 58952
        for i in range(2, 6):  # 2 / 3 / 4 / 5
            test_outputs.append(data[idkey]["lines"]["{}".format(i)]["text"])  # len(outputs): 58952


title = ['inputs', 'outputs']
train_rows = zip(train_inputs, train_outputs)
train_data_path = '../data/n_to_1/train.csv'
with open(train_data_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(title)
    for row in train_rows:
        writer.writerow(row)

dev_rows = zip(dev_inputs, dev_outputs)
dev_data_path = '../data/n_to_1/valid.csv'
with open(dev_data_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(title)
    for row in dev_rows:
        writer.writerow(row)

test_rows = zip(test_inputs, test_outputs)
test_data_path = '../data/n_to_1/test.csv'
with open(test_data_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(title)
    for row in test_rows:
        writer.writerow(row)