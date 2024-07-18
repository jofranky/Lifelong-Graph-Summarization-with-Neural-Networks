import argparse
import configparser
import site
import sys
import pickle


site.addsitedir('../../lib')  # Always appends to end

print(sys.path)


from config_utils import config_util as cfg_u

from graph_summary_generator import summary as gsg

import pathlib

import ast
import os

import numpy as np
import matplotlib.pyplot as plt



def main():
    #create graph Info
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str,dest='config', help='path to config file')  
    args = parser.parse_args()

    cfg = configparser.ConfigParser()
    cfg.read(args.config)
       
    basedir = cfg["GraphSummary"]["basedir"]
    summaries = cfg["GraphSummary"]["Summaries"]
   
    names = cfg["WorkParameter"]["names"]
    fig, ax = plt.subplots(figsize=(7, 4))
    for n in range(1,int(names)+1):
        summary = cfg["WorkParameter"]["summary"+str(n)]
        model_name= cfg["WorkParameter"]["model_name"+str(n)]



        T = int(summaries)

        arr = np.zeros((T,T))
        for i in range(1,int(summaries)+1):
            c = cfg["GraphSummary"]["summary"+str(i)]
            print("i :",i)
            for j in range(1,int(summaries)+1):
                print("j :",j)
                f = open(basedir+"/"+cfg["GraphSummary"]["summary"+str(i)]+"/Tests/"+model_name+"_"+summary+"_"+cfg["GraphSummary"]["summary"+str(j)]+".txt")
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
                right = 0
                false = 0
                for k in range(0,len(y_true)):
                    if int(y_true[k]) == int(y_pred[k]):
                        right += 1
                    else:
                        false += 1
                test = '{:.3f}'.format(right/ len(y_true),3)
                arr[i-1][j-1] = float(test)


        #Forgetting
        f =  np.zeros((T,T))
        fT = np.zeros(T)
        for t in range(1,T):
            for j in range(0,t):
                fMax= -10
                for l in range(0,t):
                    diff = arr[l][j]-arr[t][j]
                    if diff > fMax:
                        fMax = diff
                    f[t][j] = fMax

            fTS = 0 
            for j in range(0,t):
                    fTS += f[t][j]
            fT[t] = (1/(t))*(fTS)
            print("FT",t+1, " = ",fT[t])



        y1=[]
        file1 = []
        for i in range(2,T+1):
            file1.append(cfg["GraphSummary"]["summary"+str(i)][5:])
            y1.append(fT[i-1])
        print(y1)
        label = ""
        color = "b"
        if model_name == "mlp" and summary == "1":
            label = "MLP (1-Hop)"
            color = "b"
        elif model_name == "mlp" and summary == "2":
            label = "MLP (2-Hop)"
            color = "b--"
        elif model_name == "graphmlp" and summary == "2":
            label = "Graph-MLP (2-Hop)"
            color = "y--"
        elif model_name == "graphsaint" and summary == "2":
            label = "GraphSAINT (2-Hop)"
            color = "r--"
        elif model_name == "graphsaint" and summary == "3":
            label = "GraphSAINT (2-Hop and Edges)"
            color = "r:"
        ax.plot(file1, y1, color, linewidth=2.0,label=label)
        
    plt.ylabel('Forgetting')
    plt.xlabel('After task')
    plt.tick_params(axis='y', which='major')
    plt.tick_params(axis='x', which='major')    
    plt.legend(loc="upper right")
    plt.savefig("forgetting/MeasuresF.pdf")
           



if __name__ == "__main__":
    main()