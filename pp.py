import pandas as pd
import numpy as np
import os
import cv2
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import torch
import matplotlib.pyplot as plt
from torch.nn.utils.rnn import pad_sequence

df = pd.read_csv('train_cleaned.csv')
train_labels = [str(word) for word in df["Text"]]
unique_chars = sorted(set(''.join(train_labels)))
char_to_idx = {char: idx for idx, char in enumerate(unique_chars)}
n_class = len(char_to_idx)


print(char_to_idx, n_class)





def preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (200, 50))
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image


def preprocess_text(text):
    tokenized_text = [char_to_idx[char] for char in text if char in char_to_idx]
    return tokenized_text

images = []
texts = []

for idx, row in df.iterrows():
    image_path = row['Image']
    image_path = os.path.join('images', image_path)
    text = row['Text']
    
    image_data = preprocess_image(image_path)
    text_data = preprocess_text(text)
    
    images.append(image_data)
    texts.append(text_data)  

    print(idx)

    if idx+1 == 100000:
        break
    

images = np.array(images, dtype=np.float32)
print(images.shape)


texts = [torch.tensor(text, dtype=torch.long) for text in texts]
texts_padded = pad_sequence(texts, batch_first=True, padding_value=-1)

X_train, X_test, y_train, y_test = train_test_split(images, texts_padded, test_size=0.2, random_state=42)

X_train_tensor = torch.tensor(X_train)
X_test_tensor = torch.tensor(X_test)

y_train_tensor = torch.tensor(y_train)
y_test_tensor = torch.tensor(y_test)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

torch.save(train_loader, 'trainloader.pt')
torch.save(test_loader, 'testloader.pt')


def show_image_and_text(image_tensor, text_tokenized):
    image_np = image_tensor.numpy().squeeze(0) 
    print(text_tokenized)
    
    idx_to_char = {idx: char for char, idx in char_to_idx.items()}
    text_decoded = ''.join([idx_to_char[idx.item()] for idx in text_tokenized if idx != -1])
    print(text_decoded)

    plt.imshow(image_np, cmap='gray')
    plt.show()

for image, text in train_loader:
    show_image_and_text(image[0], text[0]) 
    break 

