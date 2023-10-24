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
import argparse

from models import EncoderRNN, EncoderTransformer, EncoderConv1D, LinearMixingEncoder, LinearMixingDecoder, LinearMixingModel, LinearMixingSOCPredictor
from utils import remove_outliers, postProcessSpectrum, gaus, genSeedMs, fakeTrough, A, torchA, calculate_accuracy

# python3 updatedTrain.py [model (s,t,r,c1)] [epochs]] [val_ratio] [batch_size]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training script for different models")
    parser.add_argument("model", choices=["s", "c1", "r", "t"], type=str, help="Choose a model: s, c1, r, or t")
    parser.add_argument("epochs", type=int, help="Number of training epochs")
    parser.add_argument("trainval_ts", type=float, help="Train-validation split ratio")
    parser.add_argument("batch", type=int, help="Batch Size")

    args = parser.parse_args()

wandb.init(
    # set the wandb project where this run will be logged
    project="SOC_ML_Stage2",
    name="combinedTest7linearFULL"  # Change the run name
)

# Load data and initialize as before


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
del ttrrFull, ttmFull, ttfFull, ttIhat

KEndmembers = 90
NPoints = IhFullRaCAFit.shape[0]
MSpectra = 2151

# Truth-level outputs: regressed abundances from an LMM
tFs      = torch.tensor(FsFullRaCAFit.tolist()).to(device)
tMs      = torch.tensor(msFullRaCAFit.tolist()).to(device)
trhorads = torch.tensor(rrFullRaCAFit.tolist()).to(device)

# Truth-level inputs: Individual spectra
# tIs = torch.tensor(IhFullRaCAFit.tolist()).to(device)
tmsoc    = torch.tensor(sample_soc[dataIndices.astype('int')].tolist()).to(device)
tmsoc    = tmsoc / 100.
tIs      = torch.tensor(dataI[dataIndices.astype('int')].tolist()).to(device)

# Split the data into a test set (5%) and the remaining data
X_temp, X_test, y_temp, y_test = train_test_split(IhFullRaCAFit, msFullRaCAFit, test_size=0.05, random_state=42)

# Split the remaining data into training (80%) and validation (20%)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42)
# pdb.set_trace()
# Convert the numpy arrays to PyTorch tensors and move them to the appropriate device
tMs_train = torch.tensor(y_train.tolist())
tIs_train = torch.tensor(X_train.tolist())

tMs_val = torch.tensor(y_val.tolist())
tIs_val = torch.tensor(X_val.tolist())

# Create data loaders for training and validation
train_dataset = torch.utils.data.TensorDataset(tIs_train, tMs_train)
val_dataset = torch.utils.data.TensorDataset(tIs_val, tMs_val)

batch_size = args.batch  # You can adjust this as needed
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers = 8)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers = 8)

# Training settings, optimizer declarations
nepochs = args.epochs

# Set up encoder model and optimizer
try:
    if args.model == "s":
        print("Using Standard Linear Model")
        encoder_model = LinearMixingEncoder(MSpectra, KEndmembers, 512).to(device)
    elif args.model == "c1":
        print("Using 1D Conv Model")
        encoder_model = EncoderConv1D(MSpectra, KEndmembers, 32, 15).to(device) # (M, K, hidden_size, kernel_size)
    elif args.model == "r":
        print("Using RNN Model")
        encoder_model = EncoderRNN(MSpectra, KEndmembers, 64) # (M, K, hidden_size)
    elif args.model == "t":
        print("Using Transformer Model")
        # pdb.set_trace()
        encoder_model = EncoderTransformer(MSpectra, KEndmembers, 64, 4, 2) # (M, K, hidden_size, num_heads, num_layers)
    else:
        raise ValueError("Model error, please choose s (standard linear), c1 (1D conv), r (RNN), or t (Transformer)")
except ValueError as e:
    print(e)
encoder_model = encoder_model.to(device)
# pdb.set_trace()

encoder_optimizer = optim.Adam(encoder_model.parameters(), lr=0.000001, betas=(0.99, 0.999))

# Set up decoder model and optimizer
decoder_model = LinearMixingDecoder(tFs, tMs, trhorads).to(device)
decoder_optimizer = optim.Adam(decoder_model.parameters(), lr=0.000001, betas=(0.99, 0.999))

# Rest of the code, but replace 'seedEncoderModel' with 'encoder_model' and 'decoder_model'

lossTrackingEncoder = np.zeros(nepochs);
lossTrackingDecoder = np.zeros(nepochs);
lossTrackingEncoderV = np.zeros(nepochs)  # Validation losses
lossTrackingDecoderV = np.zeros(nepochs)  # Validation losses
lossTrackingDecoderLagrangeFactor = np.zeros(nepochs);
rrTracking = np.zeros(nepochs);

encoderPreds=[]
decoderPreds=[]

train_tIs, val_tIs, train_tmsoc, val_tmsoc = train_test_split(tIs, tmsoc, test_size=args.trainval_ts, random_state=42)


# Training loop
for epoch in tqdm(range(nepochs)):
    # Log rrsoc
    rrTracking[epoch] = decoder_model.rrsoc.detach().item()

    # Initialize loss variables for this epoch
    total_encoder_loss = 0.0
    total_decoder_loss = 0.0

    # Batching and training
    for batch_data in train_loader:
        # Extract batch data
        batch_tIs, batch_tmsoc = batch_data
        batch_tIs = batch_tIs.to(device)
        batch_tmsoc = batch_tmsoc.to(device)

        # Get abundance predictions from the encoder for the batch
        encoderPreds = encoder_model(batch_tIs)

        # Get spectrum predictions from the decoder for the batch
        decoderPreds = decoder_model(encoderPreds)
        # pdb.set_trace()
        # Compute encoder loss: sqerr from true Msoc values for the batch
        encoder_loss = 1000 * torch.mean((encoderPreds[:, -1] - batch_tmsoc[:, -1]) ** 2)

        # Add decoder loss: sqerr from true RaCA spectra for the batch
        decoder_loss = torch.mean((decoderPreds - batch_tIs) ** 2)

        # Multiply decoder loss by the Lagrange factor
        decoder_loss = decoder_loss * decoder_model.computeLagrangeLossFactor()

        # Calculate the combined loss
        loss = encoder_loss + decoder_loss

        # Backpropagate the gradients for both models
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()
        loss.backward()
        encoder_optimizer.step()
        decoder_optimizer.step()

        # Accumulate batch losses
        total_encoder_loss += encoder_loss.item()
        total_decoder_loss += decoder_loss.item()

    # Calculate the average loss for this epoch
    avg_encoder_loss = total_encoder_loss / len(train_loader)
    avg_decoder_loss = total_decoder_loss / len(train_loader)

    # Log and print the average losses
    wandb.log({"Encoder_Training Loss": avg_encoder_loss, "Decoder_Training Loss": avg_decoder_loss})
    print("Epoch {}: Encoder Loss: {:.4f}, Decoder Loss: {:.4f}".format(epoch, avg_encoder_loss, avg_decoder_loss))

    # Validation Loss
    with torch.no_grad():
        # Similar batching process for validation data
        total_encoder_lossV = 0.0
        total_decoder_lossV = 0.0

        for batch_dataV in val_loader:
            batch_val_tIs, batch_val_tmsoc = batch_dataV
            batch_val_tIs = batch_val_tIs.to(device)
            batch_val_tmsoc = batch_val_tmsoc.to(device)
            encoderPredsV = encoder_model(batch_val_tIs)
            decoderPredsV = decoder_model(encoderPredsV)
            encoder_lossV = 1000 * torch.mean((encoderPredsV[:, -1] - batch_val_tmsoc[:, -1]) ** 2)
            decoder_lossV = torch.mean((decoderPredsV - batch_val_tIs) ** 2)
            total_encoder_lossV += encoder_lossV.item()
            total_decoder_lossV += decoder_lossV.item()

        avg_encoder_lossV = total_encoder_lossV / len(val_loader)
        avg_decoder_lossV = total_decoder_lossV / len(val_loader)

        wandb.log({"Encoder_Validation Loss": avg_encoder_lossV, "Decoder_Validation Loss": avg_decoder_lossV})
        print("Validation - Encoder Loss: {:.4f}, Decoder Loss: {:.4f}".format(avg_encoder_lossV, avg_decoder_lossV))

wandb.finish()