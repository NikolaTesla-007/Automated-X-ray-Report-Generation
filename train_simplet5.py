# -*- coding: utf-8 -*-
"""
train_simpleT5.ipynb
author: V Subrahmamya Raghu Ram Kihore Parupudi
"""

# installing the package
# pip install simplet5

# import and mount drive
# from google.colab import drive
# drive.mount('/content/drive')


# import torch and empty cache
import torch
torch.cuda.empty_cache()

# import required packages
import pandas as pd
from sklearn.model_selection import train_test_split
from simplet5 import SimpleT5


def load_dataset(path):
  '''load and return dataset'''
  data = pd.read_csv(path)
  return data

def remove_quotations_in_cols(data,cols=('real','prediction')):
  '''basic preprocessing (removing quotation marks)'''
  for index, row in data.iterrows():
    real = row[cols[0]]
    pred = row[cols[1]]
    processed_real = real.replace('"', '')
    processed_pred = pred.replace('"', '')
    data.at[index, cols[0]] = processed_real
    data.at[index, cols[1]] = processed_pred
  return data

def save_data_as_csv(df,save_path):
  '''save useful data as csv'''
  df.to_csv(save_path)

def split_data_into_train_val_test(df,val_by_train_fraction=0.2,test_fraction=0.2):
  '''
  splits the data into train, val and test sets.
  Firts splits the data into train_val and test,
  then splits train_val into train and val
  '''
  train_val_data, test_data = train_test_split(df, test_size=test_fraction)
  train_data, val_data = train_test_split(train_val_data, test_size=val_by_train_fraction)
  return train_data,val_data,test_data



if __name__ == "__main__":
  # load the dataset
  path = '/content/drive/Shared drives/DGM/codes/predictions500.csv'
  data = load_dataset(path)

  # remove quotation marks in text
  data = remove_quotations_in_cols(data,cols=('real','prediction'))

  # save the pre-processed data
  save_data_as_csv(data,'/content/drive/Shared drives/DGM/codes/processed_predictions500.csv')

  # simpleT5 expects dataframe to have 2 columns: "source_text" and "target_text"
  data = data.rename(columns={"real":"target_text", "prediction":"source_text"})
  data = data[['source_text', 'target_text']]

  # T5 model expects a task related prefix: since it is a summarization task, we will add a prefix "summarize: "
  data['source_text'] = "summarize: " + data['source_text']

  # get train, val and test sets
  train_data, val_data, test_data = split_data_into_train_val_test(data,val_by_train_fraction=0.25,test_fraction=0.2)

  # load the pre-trained simpleT5 model
  model = SimpleT5()
  model.from_pretrained(model_type="t5", model_name="t5-base")

  # train the model with the data, set batch_size and max_epochs accordingly
  model.train(train_df=train_data, # pandas dataframe with 2 columns: source_text & target_text
              eval_df=val_data, # pandas dataframe with 2 columns: source_text & target_text
              source_max_token_len = 512,
              target_max_token_len = 128 ,
              batch_size = 8,
              max_epochs = 3,
              use_gpu = True,
              outputdir = '/content/drive/Shared drives/DGM/codes/trained_t5',
              early_stopping_patience_epochs = 0,
              precision = 32
              )

  # save the train, val, test sets
  save_data_as_csv(train_data,'/content/drive/Shared drives/DGM/codes/train.csv')
  save_data_as_csv(val_data,'/content/drive/Shared drives/DGM/codes/val.csv')
  save_data_as_csv(test_data,'/content/drive/Shared drives/DGM/codes/test.csv')

  # load the trained model for inferencing:
  model = SimpleT5()
  path = '/content/drive/Shared drives/DGM/codes/processed_predictions500.csv'
  data = load_dataset(path)
  model.load_model("t5","/content/drive/Shared drives/DGM/codes/trained_t5/simplet5-epoch-2-train-loss-2.0227-val-loss-1.8763", use_gpu=True)
  data['improved_report'] = 0

  # for ecah prediction, improve the report
  for index, row in data.iterrows():
    input_report = row['prediction']
    improved_report = model.predict(input_report)
    data.at[index, 'improved_report'] = improved_report

  # save final results
  save_data_as_csv(data,'/content/drive/Shared drives/DGM/codes/final_results.csv')