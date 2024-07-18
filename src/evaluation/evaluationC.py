import argparse
import configparser
import site
import sys
import pickle
"""
#!/bin/bash
#SBATCH --partition=gpu_4 
#SBATCH --ntasks=1
#SBATCH --time=12:00:00
#SBATCH --mem=40000
#SBATCH --gres=gpu:1



python trainValidate.py --config config_trainValidate074.ini
"""
site.addsitedir('../../lib')  # Always appends to end

print(sys.path)

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from config_utils import config_util as cfg_u

from graph_summary_generator import summary as gsg

import pathlib

import ast
import os

import numpy as np
import matplotlib.pyplot as plt

import matplotlib as mpl

def main():
    #create graph Info
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str,dest='config', help='path to config file')  
    args = parser.parse_args()

    cfg = configparser.ConfigParser()
    cfg.read(args.config)
       
    basedir = cfg["GraphSummary"]["basedir"]
    summaries = cfg["GraphSummary"]["Summaries"]
   
    summary = cfg["WorkParameter"]["summary"]
    model_name= cfg["WorkParameter"]["model_name"]

    min = 1
    max = 0
    
    fileTr = []
    fileTe = []
    accuracies = []
    filesOrder = []
    T = int(summaries)
    arr = np.zeros((T,T))
    for i in range(1,int(summaries)+1):
        c = cfg["GraphSummary"]["summary"+str(i)]
        filesOrder.append(cfg["GraphSummary"]["summary"+str(i)][5:])
        print("i :",i)
        for j in range(1,int(summaries)+1):
            fileTr.append(cfg["GraphSummary"]["summary"+str(i)][5:])
            fileTe.append(cfg["GraphSummary"]["summary"+str(j)][5:])
            print("j :",j)
            f = open(basedir+"/"+cfg["GraphSummary"]["summary"+str(i)]+"/Tests/"+model_name+"_"+summary+"_"+cfg["GraphSummary"]["summary"+str(j)]+".txt")
            y_true = []
            y_pred = []
            #test = f.readlines()[-1].replace(" Test: ","")
            part1 = False
            lines = f.readlines()

            for line in lines[:-1]:
                if "Predicted :tensor([" in line:
                    line = line.replace("Predicted :tensor([","")
                    pred = True
                if "Real :tensor([" in line:
                    line = line.replace("Real :tensor([","")
                    pred = False
                line = line.replace("],","")
                line = line.replace(" device='cuda:0')","")
                line = line.replace("\n","")
                values = line.split(",")
                for v in values:
                    if not v.isspace() and len(v) > 0:
                        if  pred:
                            y_pred.append(int(v)) 
                        else:
                            y_true.append(int(v)) 
            
            right = 0
            false = 0
            for k in range(0,len(y_true)):
                if int(y_true[k]) == int(y_pred[k]):
                    right += 1
                else:
                    false += 1
            test = right/ len(y_true)
            if test > max:
                max = test
            if test < min:
                min = test

            accuracies.append(test) 
    dv = pd.DataFrame({'Tested on': fileTe,
                   'Trained up to': fileTr,
                   'Accuracy': accuracies})
    dvs = dv.pivot(index="Trained up to", columns="Tested on", values="Accuracy")
    dvs = dvs.reindex(filesOrder, level=0).T.reindex(filesOrder).T
    # Draw a heatmap with the numeric values in each cell
    f, ax = plt.subplots(figsize=(8, 8))
    #cmap = mpl.colormaps.get_cmap("Blues")
    #cmap.set_bad("white")
    sns.heatmap(dvs , annot=True, linewidths=.2, cmap = sns.color_palette("RdYlGn", 20),vmin=0, vmax=1,fmt='.2f')
    f = open("heatmaps/accuracy_"+model_name+"_"+summary+".txt","w+")
    f.write(str(dvs))
    f.write("\n\nmin:"+str(min)+"\nmax"+str(max))
    f.close()
    f = open("heatmaps/accuracy_"+model_name+"_"+summary+"_list.txt","w+")
    f.write(str(accuracies))
    f.close()
    plt.savefig("heatmaps/accuracy_"+model_name+"_"+summary+".pdf", bbox_inches = "tight")
    



if __name__ == "__main__":
    main()