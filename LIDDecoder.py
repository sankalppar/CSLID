import numpy as np
import torch
import glob
from tqdm import tqdm

class AdapterHead(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dense1 = torch.nn.Linear(1024, 512)
        self.act1 = torch.nn.ReLU()
        self.dense2 = torch.nn.Linear(512, 256)
        self.act2 = torch.nn.ReLU()
        self.dense3 = torch.nn.Linear(256, 6)
    def forward(self, x):
        x = self.act1(self.dense1(x))
        x = self.act2(self.dense2(x))
        x = self.dense3(x)
        return x

class EmbeddingDataset(torch.utils.data.Dataset):
    def __init__(self, emb_paths, labels_path):
        self.emb_paths = emb_paths
        self.labels_path = labels_path

    def __len__(self):
        return len(self.emb_paths)

    def __getitem__(self, idx):
        emb = torch.flatten(torch.mean(torch.load(self.emb_paths[idx]), 1))
        label = torch.load(self.labels_path + '/' + self.emb_paths[idx].split('/')[-1])
        return emb, label

if __name__ == "__main__":
    np.random.seed(0)
    torch.manual_seed(0)
    train_data_path = 'embeddings/train'
    val_data_path = 'embeddings/dev'
    train_labels_path = 'labels/train'
    val_labels_path = 'labels/dev'

    emb_paths_train = glob.glob(train_data_path + '/*')
    emb_paths_val = glob.glob(val_data_path + '/*')

    train_emb_ds = EmbeddingDataset(emb_paths_train, train_labels_path)
    val_emb_ds = EmbeddingDataset(emb_paths_val, val_labels_path)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    model = AdapterHead()
    criterion = torch.nn.KLDivLoss(reduction="batchmean")
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

    params = {'batch_size': 32,
              'shuffle': True}
    num_epochs = 10

    emb_generator = torch.utils.data.DataLoader(train_emb_ds, **params)
    emb_generator_val = torch.utils.data.DataLoader(val_emb_ds, **params)

    model = model.to(device)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch, labels in emb_generator:
            batch, labels = batch.to(device), labels.to(device)
            labels_pred = model(batch)
            loss = criterion(torch.nn.functional.log_softmax(labels_pred, dim=0), labels)
            total_loss = total_loss + loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        total_loss_val = 0
        for batch, labels in emb_generator_val:
            batch, labels = batch.to(device), labels.to(device)
            labels_pred = model(batch)
            loss = criterion(torch.nn.functional.log_softmax(labels_pred, dim=0), labels)
            total_loss_val = total_loss_val + loss.item()

        print(f'Epoch {epoch+1}/{num_epochs}, loss : {round(total_loss/len(emb_generator),5)}, loss_val : {round(total_loss_val/len(emb_generator_val),5)}')
    for i in range(10):
        emb, label = val_emb_ds[i]
        print(model(emb))
        print(label)

        
