import import_ipynb
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.stats import gaussian_kde
import torch
import torch.nn as nn
import torch.optim as optim
import json
import gc
from tqdm.notebook import tqdm
from sklearn.model_selection import train_test_split
import pdb
import pickle

from models import *
from utils import *

(XF, dataI, sample_soc, sample_socsd) = torch.load('../RaCA-data-first100.pt')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
with open('../step5_first100_E10k_lr0p00002_b1_0p99_b2_0p999.pkl', 'rb') as file:
        (model,_,_,_,_,_,dataIndices,msoc,_,_,_) = pickle.load(file)
ttrrFull = torch.cat((model.rhorad,model.rrsoc))
ttmFull  = torch.cat((model.ms,torch.tensor(msoc.tolist()).unsqueeze(1)),dim=1)
ttmFull  = (ttmFull.t() / torch.sum(ttmFull,axis=1)).t()
ttfFull  = torch.cat((model.fs,model.fsoc.unsqueeze(0)),dim=0)
ttIhat   = torch.matmul(torchA(ttmFull,ttrrFull).float(),ttfFull.float())

rrFullRaCAFit = ttrrFull.detach().numpy()
msFullRaCAFit = ttmFull.detach().numpy()
FsFullRaCAFit = ttfFull.detach().numpy()
IhFullRaCAFit = ttIhat.detach().numpy()
del ttrrFull, ttmFull, ttfFull, ttIhat, model

plt.plot(XF, dataI[dataIndices][0])
plt.plot(XF, IhFullRaCAFit[0])
plt.show()

KEndmembers = 90
NPoints = IhFullRaCAFit.shape[0]
MSpectra = 2151

# Truth-level outputs: regressed abundances from an LMM
tMs  = torch.tensor(msFullRaCAFit.tolist()).to(device)

# Truth-level inputs: Individual spectra
tIs = torch.tensor(IhFullRaCAFit.tolist()).to(device)

# Split your data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(IhFullRaCAFit, msFullRaCAFit, test_size=0.2, random_state=42)

# Convert the numpy arrays to PyTorch tensors and move them to the appropriate device
tMs_train = torch.tensor(y_train.tolist()).to(device)
tIs_train = torch.tensor(X_train.tolist()).to(device)

tMs_val = torch.tensor(y_val.tolist()).to(device)
tIs_val = torch.tensor(X_val.tolist()).to(device)

# Create data loaders for training and validation
train_dataset = torch.utils.data.TensorDataset(tIs_train, tMs_train)
val_dataset = torch.utils.data.TensorDataset(tIs_val, tMs_val)

batch_size = 32  # You can adjust this as needed
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Training settings, optimizer declarations
nepochs = 2000
seedEncoderModel = LinearMixingEncoder(MSpectra, KEndmembers, 512).to(device)
optimizer = optim.Adam(seedEncoderModel.parameters(), lr = 0.000005, betas=(0.99,0.999))
preds = []

model_parameters = filter(lambda p: p.requires_grad, seedEncoderModel.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print(params)

# Optimizer tracking
cEpoch = 0
lossTracking = np.zeros(nepochs);

# for epoch in tqdm(range(nepochs)):
#     preds = seedEncoderModel(tIs)
#     loss = (preds - tMs)**2
#     e = torch.mean(loss)
#     e.backward()
#     optimizer.step()
#     optimizer.zero_grad()

#     lossTracking[cEpoch] = e.detach().item()
#     cEpoch += 1

#     # Print loss every 100 epochs
#     if epoch % 100 == 0:
#         print(f'Epoch [{epoch}/{nepochs}], Loss: {e.item()}')

for epoch in tqdm(range(nepochs)):
    seedEncoderModel.train()  # Set the model to training mode
    for batch in train_loader:
        tIs_batch, tMs_batch = batch
        preds = seedEncoderModel(tIs_batch)
        loss = (preds - tMs_batch)**2
        e = torch.mean(loss)
        e.backward()
        optimizer.step()
        optimizer.zero_grad()

    lossTracking[cEpoch] = e.detach().item()
    cEpoch += 1

    # Print loss every 100 epochs
    if epoch % 100 == 0:
        print(f'Epoch [{epoch}/{nepochs}], Loss: {e.item()}')

    # Validation loss
    if epoch % 100 == 0:
        seedEncoderModel.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            val_loss = 0.0
            for val_batch in val_loader:
                tIs_val_batch, tMs_val_batch = val_batch
                val_preds = seedEncoderModel(tIs_val_batch)
                val_loss += torch.mean((val_preds - tMs_val_batch)**2)
            val_loss /= len(val_loader)
        print(f'Validation Loss: {val_loss.item()}')

print("Epoch ",epoch,": ", lossTracking[-1], lossTracking[-1] / (0.01 ** 2) / (KEndmembers))

# Validation loss
with torch.no_grad():
    val_preds = seedEncoderModel(tIs_val)
    val_loss = torch.mean((val_preds - tMs_val)**2)
print(f'Validation Loss: {val_loss.item()}')

_, axarr = plt.subplots(1,2,figsize=(10,5))

axarr[0].scatter(msFullRaCAFit[:,-1],np.array(preds[:,-1].tolist()))
axarr[0].set_xlabel("True SOC abundance")
axarr[0].set_ylabel("Predicted SOC abundance")

axarr[1].scatter([i for i in range(lossTracking.shape[0])],lossTracking)
axarr[1].set_xlabel("Epoch")
axarr[1].set_ylabel("chi2/NDF")

plt.tight_layout()
plt.show()

preds = seedEncoderModel(torch.tensor(dataI.tolist()).to(device))

fig, ax = plt.subplots(figsize=(5,5))
plt.scatter(sample_soc/100.,np.array(preds[:,-1].tolist()))
plt.xlabel("True SOC abundance");
plt.ylabel("Predicted SOC abundance");
plt.xlim([0,1])
plt.ylim([0,1])
plt.grid()













