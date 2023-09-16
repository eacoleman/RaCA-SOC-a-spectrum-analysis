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

plt.rcParams['text.usetex'] = True

from utils import postProcessSpectrum, gaus, genSeedMs, fakeTrough, A, torchA, calculate_accuracy
from models import LinearMixingModel

(XF, dataI, sample_soc, sample_socsd) = torch.load('../RaCA-data-first100.pt')
for iSpec in tqdm(range(dataI.shape[0])): # This is data normalization? 
            
    wavelengths = [x for x in range(350,2501)]
    reflectance = dataI[iSpec,:]
    
    newwave = np.array([wavelengths[i] for i in range(len(wavelengths)) if reflectance[i] is not None and reflectance[i] > 0.0 and reflectance[i] <= 1.0])
    newref  = np.array([reflectance[i] for i in range(len(reflectance)) if reflectance[i] is not None and reflectance[i] > 0.0 and reflectance[i] <= 1.0])
    
    dataI[iSpec,:] = postProcessSpectrum(newwave,XF,newref)

KEndmembers = 90
NPoints = dataI.shape[0]
NData = dataI.shape[0]
MSpectra = 2151

# load JSON file with pure spectra
endMemMap = json.load(open('../../data/endmember spectral data.json')) #compound values at wavelengths


# get reflectance spectra (y axis) and wavelength grid (x axis)
endMemList = [x for x in endMemMap.keys()];
endMemList.remove("General")
XF = endMemMap["General"]["Postprocessed Wavelength Axis [nm]"] #makes actual XF, other is same
print(len(XF))
F = [endMemMap[x]["Postprocessed Reflectance"] for x in endMemList] # Reflectance Values? Contains spectra of pure samples

# get density, radius info and merge into relevant arrays
rhos = [endMemMap[x]["Density (Mg/m^3)"] for x in endMemList] # Densities
rads = [endMemMap[x]["Effective Radius (nm)"] for x in endMemList] # Radius'


# Plotting SOC values

dataIndices = np.random.choice(NData,NPoints,replace=False) # NPoints number of indices from 0 to NData - 1
msoc = sample_soc[dataIndices] # random subset of the sample_soc

plt.hist(sample_soc, bins=50, density=True, alpha=0.4, label='SOC distribution from all datapoints')
plt.hist(msoc, bins=50, density=True, alpha=0.4, color='orange', label='Bootstrap SOC distribution')
plt.xlabel("SOC mass /%",fontsize=15)
plt.ylabel("Pedon count",fontsize=15)
plt.title("SOC distribution in data",fontsize=20)
plt.xscale('log')
plt.xlim([0,35])
plt.legend()
plt.grid()
plt.savefig('1SOC_Dist')
plt.close()
msoc=msoc/100.0


seedMs, seedMsDict = genSeedMs(endMemList, endMemMap, NPoints, KEndmembers, msoc)

#Reference
organicTroughs = [1650,1100,825,2060,1500,1000,751,1706,1754,1138,1170,853,877,1930,1449,2033,1524,2275,1706,1961,2137,2381,1400,1900,1791,2388]

tFsoc = np.sum(dataI.T * sample_soc,axis=1)/np.sum(sample_soc) # should only use the train for seed
trueFsoc = tFsoc - 0.125*np.sum(fakeTrough((np.zeros([1,MSpectra])+XF).T,np.array(organicTroughs),3000).T,axis=0)
seedFsoc = tFsoc - 0.125*gaus(1.0,0.5)*np.sum(fakeTrough((np.zeros([1,MSpectra])+XF).T,np.array(organicTroughs),3000*gaus(1.0,0.25,len(organicTroughs))).T,axis=0)

F = [endMemMap[x]["Postprocessed Reflectance"] for x in endMemList]
F = np.array(F + [seedFsoc])


# Generate many seeds for comparison
seedFsocs = np.tile(tFsoc,(100,1))

for i in range(seedFsocs.shape[0]) :
    seedFsocs[i,:] = seedFsocs[i,:] - 0.125*gaus(1.0,0.5)*np.sum(fakeTrough((np.zeros([1,MSpectra])+XF).T,np.array(organicTroughs),3000*gaus(1.0,0.25,len(organicTroughs))).T,axis=0)
    
plt.plot(XF,seedFsocs.T, 'orange')
plt.plot(XF,tFsoc.T, 'black',label="SOC-weighted data average")
plt.plot(XF,trueFsoc.T, 'red', label="Mean of seed distribution")
plt.legend()
plt.xlim([350,2500])
plt.xlabel("Wavelength [nm]",fontsize=15)
plt.ylabel("Reflectance",fontsize=15)
plt.title("SOC seed pseudospectra",fontsize=20)
plt.grid()
plt.savefig('2SOC_seed_pseudo')
plt.close()


rhorads = np.array(rhos)*np.array(rads)
trueSOCrr = np.mean(rhorads)
seedSOCrr = (np.mean(rhorads)*gaus(1.0,0.2))
seedAs = A(seedMs,np.append(rhorads,seedSOCrr))


plt.plot(XF,F.T);
plt.xlabel(r'Wavelength [nm]',fontsize=15)
plt.ylabel(r'Reflectance',fontsize=15)
plt.title(r'USGS pure endmember spectra',fontsize=20)
plt.xlim([350,2500])
plt.grid()
plt.savefig('3USGS_endmember_spec')
plt.close()


plt.plot(XF,np.dot(seedAs,F).T);
plt.xlabel(r'Wavelength [nm]',fontsize=15)
plt.ylabel(r'Reflectance',fontsize=15)
plt.title(r'Seed pseudospectra',fontsize=20)
plt.xlim([350,2500])
plt.grid()
plt.savefig('4Seed_pseudospec')
plt.close()


plt.plot(XF,dataI[dataIndices].T); 
plt.xlabel(r'Wavelength [nm]',fontsize=15)
plt.ylabel(r'Reflectance',fontsize=15)
plt.title(r'RaCA spectra',fontsize=20)
plt.xlim([350,2500])
plt.grid()
plt.savefig('5RaCA_spec')
plt.close()


f, ax = plt.subplots()
th = plt.hist(rhorads,bins=40);
plt.xlabel(r'Endmember rho times r [AU]',fontsize=15)
plt.ylabel(r'Count',fontsize=15)
plt.title(r'Distribution of particulate area:mass factors',fontsize=20)
plt.grid()
plt.vlines(trueSOCrr,ymin=0,ymax=np.max(th[0])*1.05,color='black',label=r'Pseudodata mean');
plt.vlines(seedSOCrr,ymin=0,ymax=np.max(th[0])*1.05,color='orange',label=r'Pseudodata seed');
ax.add_patch(Rectangle((trueSOCrr*0.8,0),trueSOCrr*0.4,np.max(th[0])*1.05,facecolor="orange",alpha=0.5,label=r'$\pm1\sigma$ band of seed distribution'));
plt.ylim([0,np.max(th[0])*1.05])
plt.legend()
plt.savefig('6Dist_partic_area_MASS')
plt.close()


# seed data: A[1:,:] and initial F's
tF       = torch.tensor(F[:-1,:].tolist())
tFsoc    = torch.tensor(seedFsoc.tolist())
tseedMs  = torch.tensor(seedMs[:,:-1].tolist())
tmsoc    = torch.tensor(msoc.tolist())
trhorads = torch.tensor(rhorads.tolist())
trrsoc   = torch.tensor(seedSOCrr)

# empirical data: (SOC values, reflectances, and max normalized reflectance)
ys = (tmsoc,torch.tensor(dataI[dataIndices].tolist()),torch.tensor([]))
pdb.set_trace()
nepochs = 10000
model = LinearMixingModel(tF,tFsoc,tseedMs,trhorads,trrsoc,nepochs)
optimizer = optim.Adam(model.parameters(), lr = 0.00002, betas=(0.99,0.999))


X_train, X_val, y_train, y_val = train_test_split(dataI, msoc, test_size=0.2, random_state=42)

def validate(model, X_val, y_val):
    with torch.no_grad():
        # Assuming you have defined the forward function in your model
        loss = model((torch.tensor(y_val), torch.tensor(X_val), torch.tensor([])))
    return loss.item()

pdb.set_trace()

for epoch in tqdm(range(nepochs)) :
    loss = model(ys)
    e = torch.mean(loss)
    e.backward()
    optimizer.step()
    if epoch % 100 == 0:
        print("Epoch ",epoch,": ", loss.detach().item(), model.lsq[epoch], model.lsq[epoch] / (0.01 ** 2) / (NPoints*MSpectra))

        with torch.no_grad():
            X_val_tensor = torch.tensor(X_val)
            y_val_tensor = torch.tensor(y_val)

            validation_predictions = model((y_val_tensor, X_val_tensor, torch.tensor([])))
            accuracy = calculate_accuracy(validation_predictions, y_val_tensor)
            print("Validation Accuracy: {:.4f}".format(accuracy))


    optimizer.zero_grad()






plt.plot(XF,model.fsoc.detach().numpy(),'black')
plt.plot(XF,seedFsoc,'blue')
plt.plot(XF,trueFsoc,'orange')
plt.savefig('7Fsocs')
print(model.rrsoc.detach().item(),seedSOCrr[0],trueSOCrr)


_, axarr = plt.subplots(3,2,figsize=(10,15))

axarr[0,0].scatter([i for i in range(len(model.lsq))],model.lsq / (0.01 ** 2) / (NPoints*MSpectra))
axarr[0,0].set_xlabel("Epoch")
axarr[0,0].set_ylabel("SQERR")
axarr[0,0].hlines(1,xmin=0,xmax=200000,color='r')
axarr[0,0].set_yscale("log")

axarr[0,1].scatter([i for i in range(len(model.loss))],model.loss / (0.01 ** 2) / (NPoints*MSpectra))
axarr[0,1].set_xlabel("Epoch")
axarr[0,1].set_ylabel("Full loss")
axarr[0,1].set_yscale("log")

axarr[1,0].scatter([i for i in range(len(model.bdsALoss))],model.bdsALoss) # what does this mean ?
axarr[1,0].set_xlabel("Epoch")
axarr[1,0].set_ylabel("Mass abundance issues loss")

axarr[1,1].scatter([i for i in range(len(model.bdsFLoss))],model.bdsFLoss)
axarr[1,1].set_xlabel("Epoch")
axarr[1,1].set_ylabel("F bounds issues loss")

axarr[2,0].scatter([i for i in range(len(model.diffloss1))],model.diffloss1)
axarr[2,0].set_xlabel("Epoch")
axarr[2,0].set_ylabel("First derivative loss")

axarr[2,1].scatter([i for i in range(len(model.difflossfull))],model.difflossfull)
axarr[2,1].set_xlabel("Epoch")
axarr[2,1].set_ylabel("Total diff loss")

plt.tight_layout()
plt.savefig('8Loss_Curves')
plt.close()



# clarification on mean vs black line?
f, axarr = plt.subplots(int(np.ceil(KEndmembers/3.)),3,figsize=(10,50))

curr_row = 0
index = 0

for iEndmember in tqdm(range(model.ms.detach().numpy().shape[1])):

    col = index % 3
    
    tcorrms = np.array(model.ms.tolist())
    tcorrms = (tcorrms > 0.0).astype('float32') * tcorrms
    tcorrms = (tcorrms.T / (np.sum(tcorrms,axis=1)) * (1-msoc)).T 
    
    # plot endmember distribution histogram
    th = axarr[curr_row,col].hist(tcorrms[:,iEndmember],bins=40,color='orange',alpha=0.3)
    
    # add mean and standard deviation bar overlay
    avg = np.mean(tcorrms[:,iEndmember])
    sd = np.sqrt(np.var(tcorrms[:,iEndmember]))
    axarr[curr_row,col].add_patch(Rectangle((avg-sd,0),sd*2,np.max(th[0])*1.05,facecolor="orange",alpha=0.2));
    axarr[curr_row,col].vlines(avg, ymin=0,ymax=np.max(th[0])*1.05,color='orange')
    
    # style
    axarr[curr_row,col].set_title(endMemList[iEndmember])
    axarr[curr_row,col].set_xlabel("Post-fit mass fraction")
    axarr[curr_row,col].set_ylabel("Count")
    axarr[curr_row,col].grid()
    axarr[curr_row,col].set_yscale('log')
    
    if endMemList[iEndmember] in seedMsDict :
        axarr[curr_row,col].vlines(seedMsDict[endMemList[iEndmember]],ymin=0,ymax=np.max(th[0]*1.05),color="black")

    # we have finished the current row, so increment row counter
    if col == 2 :
        curr_row += 1
    index +=1
    
    
f.tight_layout()
plt.savefig('9Endmember_Values')





