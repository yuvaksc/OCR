import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

class CRNN(nn.Module):
    def __init__(self, img_width, img_height, num_classes):
        super(CRNN, self).__init__()
        self.img_width = img_width
        self.img_height = img_height
        self.num_classes = num_classes

        self.conv_1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        self.conv_2 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.conv_3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv_4 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)

        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.4)
        self.reduce_dim = nn.Linear(1600, 64)

        self.lstm_1 = nn.LSTM(input_size=64, hidden_size=128, bidirectional=True, batch_first=True, dropout=0.25)
        self.lstm_2 = nn.LSTM(input_size=256, hidden_size=64, bidirectional=True, batch_first=True, dropout=0.25)

        self.output_dense = nn.Linear(128, num_classes+1)

    def forward(self, x):
        x = self.relu(self.conv_1(x))
        x = self.relu(self.conv_2(x))
        x = self.max_pool(x)
        x = self.relu(self.conv_3(x))
        x = self.relu(self.conv_4(x))
        x = self.max_pool(x)
        # print(f"Shape after convolution layers: {x.shape}")

        batch_size, channels, height, width = x.size()
        x = x.permute(0, 2, 3, 1)  # (batch_size, height, width, channels)
        x = x.contiguous().view(batch_size, height, width * channels)
        x = self.reduce_dim(x)

        x = self.dropout(x)

        
        x, _ = self.lstm_1(x)
        x, _ = self.lstm_2(x)
        # print(x.shape)

        x = self.output_dense(x)
        x = torch.softmax(x, dim=-1)


        return x



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CRNN(200, 50, num_classes=53).to(device) 
checkpoint = torch.load('checkpoint85.pth', map_location=device)
model.load_state_dict(checkpoint['state_dict']) 
model.eval()

test_loader = torch.load("testloader.pt") 

import pandas as pd 
df = pd.read_csv('train_cleaned.csv')

df = pd.read_csv('train_cleaned.csv')
train_labels = [str(word) for word in df["Text"]]
unique_chars = sorted(set(''.join(train_labels)))
char_to_idx = {char: idx for idx, char in enumerate(unique_chars)}
n_class = len(char_to_idx)
idx_to_char = {idx: char for char, idx in char_to_idx.items()}
idx_to_char[53] = ""


def decode_prediction(predictions, idx_to_char, label=0):
    decoded_texts = []
    for pred in predictions:
        if not label:
            pred_indices = pred.argmax(dim=-1)
        text = ''.join([idx_to_char[idx.item()] for idx in pred_indices if idx.item() != 53]) 
        decoded_texts.append(text)
    return decoded_texts


correct = 0
total = 0
# progress_bar = tqdm(test_loader, desc="Testing")

# for images, labels in progress_bar:
#     images = images.to(device)
#     labels = labels.to(device)

#     with torch.no_grad():
#         outputs = model(images)  # (batch_size, seq_len, num_classes)
#         outputs = outputs.permute(1, 0, 2)  # (seq_len, batch_size, num_classes)
        

#         # Decode predictions
#         predictions = outputs.permute(1, 0, 2)  # (batch_size, seq_len, num_classes)
#         decoded_predictions = decode_prediction(predictions, idx_to_char)

#         # Decode ground truth labels
#         decoded_labels = []
#         for idx, label in enumerate(labels):
#             label_text_decoded = ''.join([idx_to_char[idx.item()] for idx in label if idx != -1])
#             decoded_labels.append(label_text_decoded)


#         for pred, gt in zip(decoded_predictions, decoded_labels):
#             if pred == gt:
#                 correct += 1
#             total += 1

# accuracy = correct / total
# print(f"Accuracy: {accuracy * 100:.2f}%")


first_test_image, first_test_label = next(iter(test_loader))
first_test_image = first_test_image[22].unsqueeze(0).to(device) 
first_test_label = first_test_label[22]   

with torch.no_grad():
    first_output = model(first_test_image)[0]  # (1, seq_len, num_classes)
    first_output = first_output.argmax(dim=-1)
    print(first_output)
    first_decoded_pred = ''.join([idx_to_char[idx.item()] for idx in first_output if idx != 53])

first_decoded_gt = ''.join([idx_to_char[idx.item()] for idx in first_test_label if idx != -1])

plt.imshow(first_test_image.cpu().squeeze(0).squeeze(0).numpy(), cmap="gray")
plt.title(f"Prediction: {first_decoded_pred}\nGround Truth: {first_decoded_gt}")
plt.axis("off")
plt.show()
