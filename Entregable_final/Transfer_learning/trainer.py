from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torch.optim import Adam
from torch.utils.data import DataLoader
import torch
import tqdm
def train(chatData, model, optim, path, device):

    epochs = 12

    for i in tqdm.tqdm(range(epochs)):
        for X, a in chatData:
            X = X.to(device)
            a = a.to(device)
            optim.zero_grad()
            loss = model(X, attention_mask=a, labels=X).loss
            loss.backward()
            optim.step()
        torch.save(model.state_dict(), path)
        
def model_trainer(chatData, model, path):
    model.train()

    optim = Adam(model.parameters(), lr=1e-3)

    print("training .... ")
    train(chatData, model, optim, path)