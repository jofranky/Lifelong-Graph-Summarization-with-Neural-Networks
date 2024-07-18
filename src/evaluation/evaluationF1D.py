import argparse
import configparser
import site
import sys
import pickle
import numpy as np
from sklearn.metrics import f1_score

site.addsitedir('../../lib')  # Always appends to end
import numpy
print(sys.path)
from sklearn.metrics import classification_report
import scipy.stats
from config_utils import config_util as cfg_u
import seaborn as sns
from graph_summary_generator import summary as gsg

import pathlib

import ast
import os

import numpy as np
import matplotlib.pyplot as plt

def plot_classification_report(cr, summary,model_name):

    lines = cr.split('\n')
    f1s = []
    support = []
    members = []
    size = {}
    
    for line in lines[2 : (len(lines) - 5)]:
        t = line.split()
        f = float(t[-2])
        s = int(t[-1])
        if s in size:
            size[s].append(f)
        else:
            size[s] = [f]
    for s in size.keys():
        fs = size[s]
        fsum = 0
        for i in fs:
            fsum += i
        f1s.append(fsum/len(fs))
        support.append(s)
        members.append(len(fs))
    data = {'Support Size':support, 'F1 Score Avg. by Size':f1s}
    sns.relplot(x="Support Size", y="F1 Score Avg. by Size", data=data)
    plt.yscale('symlog',linthresh=0.00001)
    plt.ylim(0,1)
    plt.xscale('log')
    plt.savefig("F1/MeasuresF1Plot"+model_name+"_"+summary+".pdf")
    
    data = {'Support Size':support, 'Ńumber of Occurrences':members}
    sns.relplot(x="Support Size", y="Ńumber of Occurrences", data=data)
    plt.yscale('log')
    plt.xscale('log')
    plt.savefig("F1/MeasuresF1PlotM_"+model_name+"_"+summary+".pdf")

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


    f = open(basedir+"/"+cfg["GraphSummary"]["summary"+str(10)]+"/Tests/"+model_name+"_"+summary+"_"+cfg["GraphSummary"]["summary"+str(10)]+".txt")
    y_true = []
    y_pred = []
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
    

    cl = classification_report(y_true, y_pred)
    f.close()
    plot_classification_report(cl,summary,model_name)



if __name__ == "__main__":
    main()