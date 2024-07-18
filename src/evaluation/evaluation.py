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
   
    summary = cfg["WorkParameter"]["summary"]
    model_name= cfg["WorkParameter"]["model_name"]
    
    label = ""
    if model_name == "mlp" and summary == "1":
        label = "MLP (1-Hop)"
    elif model_name == "mlp" and summary == "2":
        label = "MLP (2-Hop)"
    elif model_name == "graphmlp" and summary == "2":
        label = "Graph-MLP (2-Hop)"
    elif model_name == "graphsaint" and summary == "2":
        label = "GraphSAINT (2-Hop)"
    elif model_name == "graphsaint" and summary == "3":
        label = "GraphSAINT (2-Hop and Edges)"
    
    f = open(basedir+"/"+cfg["GraphSummary"]["summary"+str(1)]+"/Tests/"+model_name+"_"+summary+"_"+cfg["GraphSummary"]["summary"+str(1)]+".txt")
    s = "\\begin{table}{ \\small \n"
    s +="\\begin{tabular}{"
    for j in range(1,int(summaries)+2):
        s += "|c"
    s += "|} \n"
    s += " \\hline \\backslashbox{Trained}{Tested} & "
    for j in range(1,int(summaries)+1):
            test = cfg["GraphSummary"]["summary"+str(j)]
            if j != int(summaries):
                s += test+" & "
            else:
                s += test+"\\\\ \n \\hline "
    print(s)
    print(f.readlines()[-1].replace(" Test: ",""))
    f.close()
    T = int(summaries)
    arr = np.zeros((T,T))
    for i in range(1,int(summaries)+1):
        
        c = cfg["GraphSummary"]["summary"+str(i)]
        s += c+" & "
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
            if j != int(summaries):
                s += test+" & "
            else:
                s += test+"\\\\ \n \\hline \n"
            arr[i-1][j-1] = float(test)
            f.close()
    if summary == "3":
        summary = "2E"
    s += "\\end{tabular}} \n \caption{Accuracy of "+label+"}\\label{"+model_name+summary+".}\n"
    s += "\\end{table}"    
    print(s)
    fs = open("accuracy/"+model_name+summary+".txt","w+")
    fs.write(s)
    
    #ACC
    ACCS = 0
    for i in range(0,T):
        ACCS += arr[T-1][i]
    ACC = (1/T)*(ACCS)
    print("ACC = ",ACC)
    
    #BWT
    BWTS = 0
    for i in range(0,T-1):
         BWTS += arr[T-1][i] - arr[i][i]
    BWT = (1/(T-1))*(BWTS)
    print("BWT = ",BWT)
    
    #FWT
    FWTS = 0
    for i in range(1,T):
         FWTS += arr[i-1][i] - arr[i][i]
    FWT = (1/(T-1))*(FWTS)
    print("FWT = ",FWT)
    
    #alphas
    a_ideal = 0
    for i in range(0,T):
        if arr[i][i] > a_ideal:
            a_ideal = arr[i][i]
            
    a_all = np.zeros(T)
    for i in range(1,T):
        aS = 0
        for j in range(1,T):
            aS += arr[i][j]
        a_all[i] = (1/T)*aS
        
    #Omegas
    o_baseS = 0
    for i in range(1,T):
            o_baseS += (arr[i][1]/a_ideal)
    o_base = (1/(T-1))*(o_baseS)
    print("OmegaBase = ",o_base)
    
    o_newS = 0
    for i in range(1,T):
            o_newS += arr[i][i]
    o_new = (1/(T-1))*(o_newS)
    print("OmegaNew = ",o_new)
    
    o_allS = 0
    for i in range(1,T):
            o_allS += (a_all[i]/a_ideal)
    o_all = (1/(T-1))*(o_allS)
    print("OmegaAll = ",o_all)
    
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
    
    s = "\\begin{table}{ \\small \n"
    s +="\\begin{tabular}{"
    for j in range(0,6):
        s += "|c"
    s += "|} \n"
    s += " \\hline ACC & BWT & FWT & $\\Omega_{base}$ & $\\Omega_{new}$ & $\\Omega_{all}$ \\\\ \n \hline "
    s += '{:.3f}'.format(ACC)+" & "+ '{:.3f}'.format( BWT) +" & "+ '{:.3f}'.format( FWT) +" & "+ '{:.3f}'.format(o_base) +" & "+'{:.3f}'.format(o_new,3) +" & "+ '{:.3f}'.format(o_all,3) + "\\\\ \n \hline "
    s += "\\end{tabular}} \n \caption{Measures of "+label+"}\\label{"+model_name+summary+"evaluation.}\n"
    s += "\\end{table}"   
    fs = open("measures/"+model_name+summary+"Measures.txt","w+")
    fs.write(s)
    fs.close()
    
    
    
    
    
    s2 =  "\\begin{table}{ \\small \n"
    s2 +="\\begin{tabular}{"
    for j in range(0,T-1):
        s2 += "|c"
    s2 += "|} \n \\hline "
    for j in range(0,T-1):
        s2 += "$F_{"+str(j+2)+"}$"
        if j < T-2:
            s2 += " & "
    s2 += "\\\\ \n \hline "
    for j in range(0,T-1):
        s2 += '{:.3f}'.format(fT[j+1])
        if j < T-2:
            s2 += " & "
    s2 += "\\\\ \n \hline "
    s2 += "\\end{tabular}} \n \caption{Forgetting of "+model_name+" on Summary AC\_"+summary+"}\\label{"+model_name+summary+"evaluation}\n"
    s2 += "\\end{table}"   
    fs = open("forgetting/"+model_name+summary+"MeasuresF.txt","w+")
    fs.write(s2)
    fs.close()
    
    
           



if __name__ == "__main__":
    main()
