import pandas as pd

ori_valid_path = '../data/n_to_1_new/valid.csv'
ori_test_path = '../data/n_to_1_new/test.csv'

ori_df_valid = pd.read_csv(ori_valid_path, )
ori_df_test = pd.read_csv(ori_test_path)


all_df = pd.concat([ori_df_valid, ori_df_test])  # all_df.shape: (19412, 5)

df_emo_train = all_df[:int((all_df.shape[0] * 0.8 // 4) * 4)]
df_emo_valid = all_df[int((all_df.shape[0] * 0.8 // 4) * 4):-int((all_df.shape[0] * 0.1 // 4) * 4)]
df_emo_test = all_df[-int((all_df.shape[0] * 0.1 // 4) * 4):]

df_emo_train.to_csv('data/n_to_1_new/emo_data/train.csv', index=False)
df_emo_valid.to_csv('data/n_to_1_new/emo_data/valid.csv', index=False)
df_emo_test.to_csv('data/n_to_1_new/emo_data/test.csv', index=False)