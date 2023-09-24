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
import wandb
import time
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression

from models import *
from utils import *

wandb.init(
    # set the wandb project where this run will be logged
    project="SOC_ML",
    
    # track hyperparameters and run metadata

    name = "step9_stage1"
)


# (XF, dataI, sample_soc, sample_socsd) = torch.load('../RaCA-data-first100.pt')
data = np.loadtxt("/home/sujaynair/RaCA-spectra-raw.txt", delimiter=",",dtype=str)
sample_bd = data[1:,2158].astype('float32')
sample_bdsd = data[1:,2159].astype('float32')
sample_soc = data[1:,2162].astype('float32')
sample_socsd = data[1:,2163].astype('float32')
print("Loaded txt")

XF = np.array([x for x in range(350,2501)]);

with open('/home/sujaynair/RaCA-data.pkl', 'rb') as file:
    dataI = pickle.load(file)
del data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open('/home/sujaynair/RaCA-SOC-a-spectrum-analysis/RaCA-SOC-a/Step5/Step5FULLMODEL.pkl', 'rb') as file:
        (model,_,_,_,_,_,dataIndices,msoc,_,_,_) = pickle.load(file)

# pdb.set_trace()
ttrrFull = torch.cat((model.rhorad,model.rrsoc))
ttmFull  = torch.cat((model.ms,torch.tensor(msoc.tolist()).unsqueeze(1)),dim=1)
ttmFull  = (ttmFull.t() / torch.sum(ttmFull,axis=1)).t()
ttfFull  = torch.cat((model.fs,model.fsoc.unsqueeze(0)),dim=0)
ttIhat   = torch.matmul(torchA(ttmFull,ttrrFull).float(),ttfFull.float())
# pdb.set_trace()
rrFullRaCAFit = ttrrFull.detach().numpy()
msFullRaCAFit = ttmFull.detach().numpy()
FsFullRaCAFit = ttfFull.detach().numpy()
IhFullRaCAFit = ttIhat.detach().numpy()
del model

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
tMs_train = torch.tensor(y_train.tolist())
tIs_train = torch.tensor(X_train.tolist())

tMs_val = torch.tensor(y_val.tolist())
tIs_val = torch.tensor(X_val.tolist())

# Create data loaders for training and validation
train_dataset = torch.utils.data.TensorDataset(tIs_train, tMs_train)
val_dataset = torch.utils.data.TensorDataset(tIs_val, tMs_val)

batch_size = 32  # You can adjust this as needed
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers = 8)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers = 8)

# Training settings, optimizer declarations
nepochs = 50
seedEncoderModel = LinearMixingEncoder(MSpectra, KEndmembers, 512).to(device)
optimizer = optim.Adam(seedEncoderModel.parameters(), lr = 0.000005, betas=(0.99,0.999))
preds = []

model_parameters = filter(lambda p: p.requires_grad, seedEncoderModel.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print(params)

# Optimizer tracking
cEpoch = 0
lossTracking = np.zeros(nepochs);



training_losses = []
validation_losses = []
for epoch in tqdm(range(nepochs)): # make train val curves
    seedEncoderModel.train()  # Set the model to training mode
    t3 = time.time()
    for batch in train_loader:
        tIs_batch, tMs_batch = batch
        tIs_batch = tIs_batch.to(device)
        tMs_batch = tMs_batch.to(device)
        t1 = time.time()
        preds = seedEncoderModel(tIs_batch)
        # pdb.set_trace()
        t2 = time.time()
        loss = (preds - tMs_batch)**2
        e = torch.mean(loss)
        e.backward()
        optimizer.step()
        optimizer.zero_grad()
        t3 = time.time()
        training_losses.append(e.item())

    lossTracking[cEpoch] = e.detach().item()
    cEpoch += 1

    # Print loss every 100 epochs
    if epoch % 100 == 0:
        print(f'Epoch [{epoch}/{nepochs}], Loss: {e.item()}')

    # Validation loss
    # if epoch % 100 == 0:
    seedEncoderModel.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        val_loss = 0.0
        for val_batch in val_loader:
            tIs_val_batch, tMs_val_batch = val_batch
            tIs_val_batch = tIs_val_batch.to(device)
            tMs_val_batch = tMs_val_batch.to(device)
            val_preds = seedEncoderModel(tIs_val_batch)
            val_loss += torch.mean((val_preds - tMs_val_batch)**2)
        val_loss /= len(val_loader)
        validation_losses.append(val_loss.item())
    print(f'Validation Loss: {val_loss.item()}')

    wandb.log({"Training Loss": e.item(), "Validation Loss": val_loss.item(), "Batch Processing": t1-t3, "Forward Pass": t2-t1, "Backward Pass": t3-t2})
wandb.finish()
epochs = np.arange(1, nepochs + 1)

# Plot training and validation losses on the same plot
# plt.figure(figsize=(10, 6))
# plt.plot(epochs, lossTracking, 'b', label='Training Loss')
# plt.plot(epochs[::100], validation_losses, 'r', label='Validation Loss')
# plt.title('Training and Validation Loss Curves')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.grid(True)
# plt.savefig('train_val_curves')
preds = seedEncoderModel(tIs)
# pdb.set_trace()

_, axarr = plt.subplots(1,2,figsize=(10,5))

axarr[0].scatter(msFullRaCAFit[:,-1],np.array(preds[:,-1].tolist()))
axarr[0].set_xlabel("True SOC abundance")
axarr[0].set_ylabel("Predicted SOC abundance")

axarr[1].scatter([i for i in range(lossTracking.shape[0])],lossTracking)
axarr[1].set_xlabel("Epoch")
axarr[1].set_ylabel("chi2/NDF")

plt.tight_layout()
plt.show()

# pdb.set_trace()

preds = seedEncoderModel(torch.tensor(dataI.tolist()).to(device))
# pdb.set_trace()
fig, ax = plt.subplots(figsize=(5,5))
plt.scatter(sample_soc/100.,np.array(preds[:,-1].tolist()))
plt.xlabel("True SOC abundance");
plt.ylabel("Predicted SOC abundance");
plt.xlim([0,1])
plt.ylim([0,1])
plt.grid()


rrFullRaCAFit = ttrrFull.detach().numpy()
msFullRaCAFit = ttmFull.detach().numpy()
FsFullRaCAFit = ttfFull.detach().numpy()
IhFullRaCAFit = ttIhat.detach().numpy()

del ttrrFull, ttmFull, ttfFull, ttIhat

# Model seeds: Fs, Ms, and rhorads
tFs      = torch.tensor(FsFullRaCAFit.tolist()).to(device)
tMs      = torch.tensor(msFullRaCAFit.tolist()).to(device)
trhorads = torch.tensor(rrFullRaCAFit.tolist()).to(device)

# Truth-level data:
tmsoc    = torch.tensor(sample_soc[dataIndices.astype('int')].tolist()).to(device)
tmsoc    = tmsoc / 100.
tIs      = torch.tensor(dataI[dataIndices.astype('int')].tolist()).to(device)

preds = seedEncoderModel(tIs)

def remove_outliers(data, threshold=3):
    z_scores = np.abs((data - data.mean()) / data.std())
    return data[z_scores < threshold]

# pdb.set_trace()
# Convert PyTorch tensors to NumPy arrays if needed
tmsoc = tmsoc.cpu().detach().numpy()
preds = preds[:, -1].cpu().detach().numpy()

# Remove major outliers from the data
tmsoc_no_outliers = remove_outliers(tmsoc)
preds_no_outliers = remove_outliers(preds)

# Ensure both tmsoc_no_outliers and preds_no_outliers have the same number of samples
min_samples = min(len(tmsoc_no_outliers), len(preds_no_outliers))
tmsoc_no_outliers = tmsoc_no_outliers[:min_samples]
preds_no_outliers = preds_no_outliers[:min_samples]

# Fit a linear regression model to the data without outliers
lr = LinearRegression().fit(tmsoc_no_outliers.reshape(-1, 1), preds_no_outliers.reshape(-1, 1))

# Predict values using the linear model
preds_lr = lr.predict(tmsoc_no_outliers.reshape(-1, 1))

# Calculate R^2 value for the data without outliers
r_squared = r2_score(preds_no_outliers, preds_lr)

# Create a scatter plot for the data without outliers
plt.figure(figsize=(10, 6))
plt.scatter(tmsoc_no_outliers, preds_no_outliers, s=10, label='Data Points (No Outliers)')

# Plot the linear regression line for the data without outliers
plt.plot(tmsoc_no_outliers, preds_lr, color='red', linewidth=2, label='Linear Regression Line')

# Format the R^2 value as plain text
r_squared_text = r'$R^2 = {:.2f}$'.format(r_squared)

# Add R^2 value to the plot
plt.text(0.5, 0.9, r_squared_text, fontsize=12, transform=plt.gca().transAxes)

# Add labels and a legend
plt.xlabel('Ground Truth msoc (tmsoc)')
plt.ylabel('Predicted msoc (preds[:, 1])')
plt.legend()

# Display the plot
plt.title('Scatter Plot of Predicted vs. Ground Truth msoc (No Major Outliers)')
plt.grid(True)
plt.show()






tmsoc    = torch.tensor(sample_soc[dataIndices.astype('int')].tolist()).to(device)
tmsoc    = tmsoc / 100.

nepochs = 3000 #1000000
encoderModel = seedEncoderModel.to(device)
decoderModel = LinearMixingDecoder(tFs, tMs, trhorads).to(device)
encoderDecoderParams = list(decoderModel.parameters()) + list(encoderModel.parameters())
optimizer = optim.Adam(encoderDecoderParams, lr = 0.000001, betas=(0.99,0.999))


encoderModelParams = filter(lambda p: p.requires_grad, encoderModel.parameters())
numEncoderParams = sum([np.prod(p.size()) for p in encoderModelParams])
print(numEncoderParams)

decoderModelParams = filter(lambda p: p.requires_grad, decoderModel.parameters())
numDecoderParams = sum([np.prod(p.size()) for p in decoderModelParams])
print(numDecoderParams)
# pdb.set_trace()
# Optimizer tracking
cEpoch = 0
lossTrackingEncoder = np.zeros(nepochs);
lossTrackingDecoder = np.zeros(nepochs);
lossTrackingEncoderV = np.zeros(nepochs)  # Validation losses
lossTrackingDecoderV = np.zeros(nepochs)  # Validation losses
lossTrackingDecoderLagrangeFactor = np.zeros(nepochs);
rrTracking = np.zeros(nepochs);

encoderPreds=[]
decoderPreds=[]

train_tIs, val_tIs, train_tmsoc, val_tmsoc = train_test_split(tIs, tmsoc, test_size=0.2, random_state=42)


# for epoch in tqdm(range(nepochs)) :
#     # Log rrsoc
#     rrTracking[cEpoch] = decoderModel.rrsoc.detach().item()

#     # Get abundance predictions from encoder
#     encoderPreds = encoderModel(tIs)
    
#     # Get spectrum predictions from decoder
#     decoderPreds = decoderModel(encoderPreds)
    
#     # Compute encoder loss: sqerr from true Msoc values
#     loss = 1000*torch.mean((encoderPreds[:,-1] - tmsoc)**2) 
    
    
#     [cEpoch] = loss.detach().item()
    
#     # Add decoder loss: sqerr from true RaCA spectra
#     dcLoss = torch.mean((decoderPreds - tIs)**2)
#     lossTrackingDecoder[cEpoch] = dcLoss.detach().item()
#     lossTrackingDecoderLagrangeFactor[cEpoch] = decoderModel.computeLagrangeLossFactor().detach().item()
#     dcLoss = dcLoss * decoderModel.computeLagrangeLossFactor()

#     loss = loss + dcLoss

#     loss.backward()

#     decoderModel.Fs.grad[:-1,:] = 0

#     optimizer.step()
#     optimizer.zero_grad()

#     cEpoch += 1


wandb.init(
    # set the wandb project where this run will be logged
    project="SOC_ML",
    
    # track hyperparameters and run metadata
    name = "step9_stage2"
)


# Training loop
for epoch in tqdm(range(nepochs)):
    print(epoch)
    # Log rrsoc
    rrTracking[epoch] = decoderModel.rrsoc.detach().item()

    # Get abundance predictions from encoder for training set
    encoderPreds = encoderModel(train_tIs)

    # Get spectrum predictions from decoder for training set
    decoderPreds = decoderModel(encoderPreds)

    # Compute encoder loss: sqerr from true Msoc values for training set
    loss = 1000 * torch.mean((encoderPreds[:, -1] - train_tmsoc) ** 2)
    lossTrackingEncoder[epoch] = loss.detach().item()

    # Add decoder loss: sqerr from true RaCA spectra for training set
    dcLoss = torch.mean((decoderPreds - train_tIs) ** 2)
    lossTrackingDecoder[epoch] = dcLoss.detach().item()
    lossTrackingDecoderLagrangeFactor[epoch] = decoderModel.computeLagrangeLossFactor().detach().item()

    # Multiply decoder loss by Lagrange factor
    dcLoss = dcLoss * decoderModel.computeLagrangeLossFactor()

    loss = loss + dcLoss

    loss.backward()

    decoderModel.Fs.grad[:-1, :] = 0

    optimizer.step()
    optimizer.zero_grad()
    wandb.log({"Encoder_Training Loss": loss.detach().item(), "Decoder_Training Loss": dcLoss.detach().item()})
    print(lossTrackingEncoder[epoch])
    print(lossTrackingDecoder[epoch])
    # Validation Loss
    with torch.no_grad():
        # Get abundance predictions from encoder for validation set
        encoderPredsV = encoderModel(val_tIs)

        # Get spectrum predictions from decoder for validation set
        decoderPredsV = decoderModel(encoderPredsV)

        # Compute encoder loss: sqerr from true Msoc values for validation set
        lossV = 1000 * torch.mean((encoderPredsV[:, -1] - val_tmsoc) ** 2)
        lossTrackingEncoderV[epoch] = lossV.item()

        # Add decoder loss: sqerr from true RaCA spectra for validation set
        dcLossV = torch.mean((decoderPredsV - val_tIs) ** 2)
        lossTrackingDecoderV[epoch] = dcLossV.item()

        wandb.log({"Encoder_Validation Loss": lossV.item(), "Decoder_Validation Loss": dcLossV.item()})

    print(lossTrackingEncoderV[epoch])
    print(lossTrackingDecoderV[epoch])

    # After each epoch, log the metrics individually with x-axis values

    
    


wandb.finish()


print("Epoch ",epoch,": ", lossTrackingEncoder[-1]+lossTrackingDecoder[-1])

# Create a single figure with four subplots
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Plot training and validation loss for encoder
axes[0, 0].scatter(range(lossTrackingEncoder.shape[0]), lossTrackingEncoder, label='Train')
axes[0, 0].set_xlabel("Epoch")
axes[0, 0].set_ylabel("chi2/NDF")
axes[0, 0].legend()
axes[0, 0].set_title("Encoder Training Loss")

axes[0, 1].scatter(range(lossTrackingEncoderV.shape[0]), lossTrackingEncoderV, label='Validation')
axes[0, 1].set_xlabel("Epoch")
axes[0, 1].set_ylabel("chi2/NDF")
axes[0, 1].legend()
axes[0, 1].set_title("Encoder Validation Loss")

# Plot training and validation loss for decoder
axes[1, 0].scatter(range(lossTrackingDecoder.shape[0]), lossTrackingDecoder, label='Train')
axes[1, 0].set_xlabel("Epoch")
axes[1, 0].set_ylabel("chi2/NDF")
axes[1, 0].legend()
axes[1, 0].set_title("Decoder Training Loss")

axes[1, 1].scatter(range(lossTrackingDecoderV.shape[0]), lossTrackingDecoderV, label='Validation')
axes[1, 1].set_xlabel("Epoch")
axes[1, 1].set_ylabel("chi2/NDF")
axes[1, 1].legend()
axes[1, 1].set_title("Decoder Validation Loss")

# Adjust the layout
plt.tight_layout()

# Show the plot
plt.show()


_, axarr = plt.subplots(3,2,figsize=(10,10))

axarr[0,0].scatter(np.array(tmsoc.tolist()),np.array(encoderPreds[:,-1].tolist()))
axarr[0,0].set_xlabel("True SOC abundance")
axarr[0,0].set_ylabel("Encoder-predicted SOC abundance")
axarr[0,0].set_xlim([0,1])
axarr[0,0].set_ylim([0,1])

axarr[0,1].scatter([i for i in range(cEpoch)],lossTrackingEncoder[:cEpoch])
axarr[0,1].set_xlabel("Epoch")
axarr[0,1].set_ylabel("Encoder Loss")

axarr[1,0].scatter([i for i in range(cEpoch)],lossTrackingDecoder[:cEpoch])
axarr[1,0].set_xlabel("Epoch")
axarr[1,0].set_ylabel("Decoder Loss")

axarr[2,1].scatter([i for i in range(cEpoch)],lossTrackingEncoder[:cEpoch]+lossTrackingDecoder[:cEpoch])
axarr[2,1].set_xlabel("Epoch")
axarr[2,1].set_ylabel("Total Loss")

axarr[2,0].scatter([i for i in range(cEpoch)],rrTracking[:cEpoch])
axarr[2,0].set_xlabel("Epoch")
axarr[2,0].set_ylabel("(rho*r) of SOC")

axarr[1,1].plot(XF, decoderModel.Fs.float().cpu().detach()[-1,:],color='orange',alpha=0.5)
#axarr[1,1].plot(XF, FsFullRaCAFit[-1,:],color='blue',alpha=0.6)
axarr[1,1].set_xlabel("Wavelength [nm]")
axarr[1,1].set_ylabel("Reflectance")

plt.tight_layout()
plt.show()

# load JSON file with pure spectra
endMemMap = json.load(open('/home/sujaynair/RaCA-SOC-a-spectrum-analysis/data/endmember spectral data.json'))

# get reflectance spectra (y axis) and wavelength grid (x axis)
endMemList = [x for x in endMemMap.keys()] + ["SOC"];
endMemList.remove("General")

f, axarr = plt.subplots(int(np.ceil(KEndmembers/3.)),3,figsize=(10,50))

curr_row = 0
index = 0

for iEndmember in tqdm(range(KEndmembers)):

    col = index % 3
    
    # plot endmember distribution histogram
    axarr[curr_row,col].plot(XF,decoderModel.Fs.float().cpu().detach()[iEndmember,:],color='orange',alpha=0.5)
    axarr[curr_row,col].plot(XF, FsFullRaCAFit[iEndmember,:],color='blue',alpha=0.6)
    
    # style
    axarr[curr_row,col].set_xlabel(endMemList[iEndmember])
    #axarr[curr_row,col].set_xlabel("Wavelength [nm]")
    #axarr[curr_row,col].set_ylabel("Reflectance")
    #axarr[curr_row,col].grid()


    # we have finished the current row, so increment row counter
    if col == 2 :
        curr_row += 1
    index +=1
    
    
f.tight_layout()
plt.show()








#Ecol ONvohp4gqob@
