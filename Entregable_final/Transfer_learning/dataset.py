!pip install datasets
from datasets import load_dataset
import csv
import os
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os

class Data(Dataset):
  def __init__(self, path:str, tokenizer):
      self.data = pd.io.parsers.read_csv(path)
      print(len(self.data))
      self.X=[]
      self.X.append(self.data["Input"][0])
      for i in range(1493):
        self.X.append(self.data["Output"][i])


      for idx, i in enumerate(self.X):
          try:
              self.X[idx] = "<BOS> "+i+" <bot>: "+self.X[idx+1]+" <EOS>"
          except:
              break
      self.X = self.X[:-1]



      self.X_encoded = tokenizer(self.X,max_length=40, truncation=True, padding="max_length", return_tensors="pt")
      self.input_ids = self.X_encoded['input_ids']
      self.attention_mask = self.X_encoded['attention_mask']

  def __len__(self):
      return len(self.X)

  def __getitem__(self, idx):
      return (self.input_ids[idx], self.attention_mask[idx])


def write_to_csv(conversations, output_filename):
    with open(output_filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)

        writer.writerow(['Input', 'Output'])

        writer.writerows(conversations)

# Citation:
# Li, Y., Su, H., Shen, X., Li, W., Cao, Z., & Niu, S. (2017). Dailydialog: A manually labelled multi-turn dialogue dataset. arXiv preprint arXiv:1710.03957.
def load_dataset(path, tokenizer):
    csv_path = os.path.dirname(os.path.realpath(__file__)) + "dataset.csv"
    if(!os.path.isfile(csv_path)):
        dataset = load_dataset("daily_dialog")

        train_dataset = dataset['train']
        conversations = []
        for conv in train_dataset['dialog']:
          for i in range(1, len(conv)):
            conversations.append((conv[i-1], conv[i]))
            
        
        write_to_csv(conversations, csv_path)
    chatData = Data(csv_path, tokenizer)
    chatData =  DataLoader(chatData, batch_size=64)
    return chatData
