import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

files = []
for fl in os.listdir("results/analysis"):
    if ".DS_" not in fl:
        files.append(fl)

final_val = np.ones((int(10**4),4))*np.nan
for i, fl in enumerate(files):
    nm = int(fl.split(".csv")[0])
    df = pd.read_csv("results/analysis/%s"%fl)
    final_val[nm] = df.iloc[1000].values[1:]
    if (i%50) ==0:
        print(i/len(files))

final_val_grid = final_val.reshape(10,10,10,10,4)
external_score_grid = final_val_grid[...,3]/(final_val_grid[...,3] + final_val_grid[...,2])
external_score = np.nanmean(external_score_grid,axis=-1)
final_val_grid_mean = np.nanmean(final_val_grid,axis=-2)
# external_score = final_val_grid_mean[...,3]/(final_val_grid_mean[...,3] + final_val_grid_mean[...,2])
fig, ax = plt.subplots()
ax.imshow(external_score[:,])
fig.show()


fig, ax = plt.subplots()
ax.imshow(external_score[:,np.arange(10),np.arange(10)])
fig.show()

all_val = np.ones((int(10**4),1001,4))*np.nan
for i, fl in enumerate(files):
    nm = int(fl.split(".csv")[0])
    df = pd.read_csv("results/analysis/%s"%fl)
    all_val[nm] = df.values[:,1:]
    if (i%50) ==0:
        print(i/len(files))

all_val_grid = all_val.reshape(10,10,10,10,1001,4)
all_external_grid = np.log10(all_val_grid[...,3]/all_val_grid[...,2])

fig, ax = plt.subplots(figsize=(4,4))
for i in range(0,10,2):
    ax.plot(all_external_grid[i,-1,-1].mean(axis=0),color=plt.cm.plasma(i/10),alpha=0.5)
for i in range(0,10,2):
    ax.plot(all_external_grid[i,0,0].mean(axis=0),color=plt.cm.plasma(i/10))
ax.set(ylabel="Externalisation index",xlabel="Time")
fig.tight_layout()
fig.show()

fig, ax = plt.subplots()
ax.imshow(all_external_grid[:,np.arange(10),np.arange(10),:,-100:].mean(axis=(-2,-1)),vmin=0,vmax=0.25)
fig.show()