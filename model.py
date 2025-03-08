import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
import jiwer

df = pd.read_csv('train_cleaned.csv')
train_labels = [str(word) for word in df["Text"]]
unique_chars = sorted(set(''.join(train_labels)))
char_to_idx = {char: idx for idx, char in enumerate(unique_chars)}
n_class = len(char_to_idx)
print(n_class)
print(char_to_idx)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

train_loader = torch.load('trainloader.pt')






class CRNN(nn.Module):
    def __init__(self, img_width, img_height, num_classes):
        super(CRNN, self).__init__()
        self.img_width = img_width
        self.img_height = img_height
        self.num_classes = num_classes

        # Convolutional layers
        self.conv_1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        self.conv_2 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.conv_3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv_4 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)

        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.4)
        self.reduce_dim = nn.Linear(1600, 64)

        # LSTM Decoder layers
        self.lstm_1 = nn.LSTM(input_size=64, hidden_size=128, bidirectional=True, batch_first=True, dropout=0.25)
        self.lstm_2 = nn.LSTM(input_size=256, hidden_size=64, bidirectional=True, batch_first=True, dropout=0.25)

        # Final dense layer
        self.output_dense = nn.Linear(128, num_classes+1)

    def forward(self, x):
        x = self.relu(self.conv_1(x))
        x = self.relu(self.conv_2(x))
        x = self.max_pool(x)
        x = self.relu(self.conv_3(x))
        x = self.relu(self.conv_4(x))
        x = self.max_pool(x)  # 32 x 1 x 50 x 200
        # 32 x n x 53

        batch_size, channels, height, width = x.size()
        x = x.permute(0, 2, 3, 1)  # (batch_size, height, width, channels)
        x = x.contiguous().view(batch_size, height, width * channels)
        x = self.reduce_dim(x)
        x = self.dropout(x)

        x, _ = self.lstm_1(x)
        x, _ = self.lstm_2(x)

        # Final output
        x = self.output_dense(x)
        x = torch.softmax(x, dim=-1)


        return x




criterion = nn.CTCLoss(blank=53, zero_infinity=True)

model = CRNN(200, 50, n_class).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)


###
checkpoint = torch.load('checkpoint85.pth', map_location=device)

model.load_state_dict(checkpoint['state_dict'])

optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

start_epoch = checkpoint['epoch'] 
avg_loss = checkpoint['avg_loss']
avg_cerp = checkpoint['avg_cerp']

print(f"Resuming from epoch {start_epoch}, Avg Loss: {avg_loss:.4f}, Avg CERP: {avg_cerp:.4f}")
###




num_epochs = 90
for epoch in range(start_epoch, num_epochs):
    model.train()
    epoch_loss = 0
    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch + 1}/{num_epochs}")
    total_cerp = 0

    for batch_idx, (images, labels) in progress_bar:
        # Move data to device
        images = images.to(device) 
        labels = labels.to(device)  

        mask = labels != -1 
        target_lengths = mask.sum(dim=1)


        outputs1 = model(images)  # Shape: (batch_size, seq_len, num_classes)

        # CTC loss (seq_len, batch_size, num_classes)
        outputs = outputs1.permute(1, 0, 2)

        input_lengths = torch.full((images.size(0),), fill_value=outputs.size(0), dtype=torch.long, device=device)  # Adjusted for downsampled input width

        criterion.ignore_index = -1
        output_mask = (outputs != 53)
        loss = criterion(outputs*output_mask, labels, input_lengths, target_lengths)

        tokens = outputs1.argmax(dim=-1)
        batch_cerp = 0

        idx_to_char = {idx: char for char, idx in char_to_idx.items()}
        idx_to_char[53] = ""

        # CER Loss
        for idx, label in enumerate(labels):
            label_text_decoded = ''.join([idx_to_char[idx.item()] for idx in label if idx != -1])
            output_text_decoded = ''.join([idx_to_char[idx.item()] for idx in tokens[idx] if idx != 53])


            output_text_decoded = output_text_decoded.strip()
            label_text_decoded = label_text_decoded.strip()
            # print(label_text_decoded, output_text_decoded)

            if not output_text_decoded or not label_text_decoded:
                cerp = 1
            else:
                cerp = jiwer.cer(output_text_decoded, label_text_decoded)
            cerp *= 100
            batch_cerp += cerp 

        batch_cerp = batch_cerp/len(labels)
        total_cerp += batch_cerp

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        progress_bar.set_postfix({"Batch Loss": loss.item(), "Avg Loss": epoch_loss / (batch_idx + 1), "CER": round(batch_cerp, 2)})

    print(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {epoch_loss / len(train_loader):.4f}, Avg Cerp: {total_cerp/len(train_loader)}")


    if epoch % 5 == 0:
        torch.save({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'avg_loss': epoch_loss / len(train_loader),
                'avg_cerp': total_cerp/len(train_loader),
            }, f'checkpoint{epoch}.pth')
