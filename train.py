from nltk_utils import tokenize, stem, bag_of_words
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch
import numpy as np
import json
import random
from model import NeuralNet

with open('data.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

all_words = []
tags = []
xy = []
for intent in data['intents']:
    tag = intent['tag']
    tags.append(tag)
    # loop through each sentence in the intent
    for pattern in intent['patterns']:
        # tokenize each word in the sentence
        word = tokenize(pattern)
        all_words.extend(word)
        # add to xy
        xy.append((word, tag))

ignore_words = ['?', '!', '.', ',', ':', ';', '(', ')', '[', ']', '{', '}', '"', '"', "'", "'"]
all_words = [stem(w) for w in all_words if w not in ignore_words]
all_words = sorted(set(all_words))
tags = sorted(set(tags))
# print(tags)

X_train = []
y_train = []
for (pattern_sentence, tag) in xy:
    # bag of words
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)
    label = tags.index(tag)
    y_train.append(label)

X_train = np.array(X_train)
y_train = np.array(y_train)

class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
    

# Hyperparameters
batch_size = 8
hidden_size = 8
output_size = len(tags)
input_size = len(X_train[0])
learning_rate = 0.001
num_epochs = 1000


dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=2)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NeuralNet(input_size, hidden_size, output_size).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(words)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

print(f'final loss: {loss.item():.4f}')


data = {
    'model_state': model.state_dict(),
    'input_size': input_size,
    'output_size': output_size,
    'hidden_size': hidden_size,
    'all_words': all_words,
    'tags': tags
}

FILE = "data.pth"
torch.save(data, FILE)
print(f'Training complete. File saved to {FILE}')