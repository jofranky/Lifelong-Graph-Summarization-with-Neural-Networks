import argparse
import configparser
import site
import sys
import pickle
import numpy as np
from sklearn.metrics import f1_score

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
    s1 = "\\begin{table}{ \\small \n"
    s1 +="\\begin{tabular}{"
    s2 = "\\begin{table}{ \\small \n"
    s2 +="\\begin{tabular}{"
    for j in range(1,int(summaries)+2):
        s += "|c"
        s1 += "|c"
        s2 += "|c"
    s += "|} \n"
    s += " \\hline \\backslashbox{Trained}{Tested} & "
    s1 += "|} \n"
    s1 += " \\hline \\backslashbox{Trained}{Tested} & "
    s2 += "|} \n"
    s2 += " \\hline \\backslashbox{Trained}{Tested} & "
    for j in range(1,int(summaries)+1):
            test = cfg["GraphSummary"]["summary"+str(j)]
            if j != int(summaries):
                s += test+" & "
                s1 += test+" & "
                s2 += test+" & "
            else:
                s += test+"\\\\ \n \\hline "
                s1 += test+"\\\\ \n \\hline "
                s2 += test+"\\\\ \n \\hline "
    print(s)
    for i in range(1,int(summaries)+1):
        c = cfg["GraphSummary"]["summary"+str(i)]
        s += c+" & "
        s1 += c+" & "
        s2 += c+" & "
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
            fma = f1_score(y_true, y_pred, average='macro')
            fmi = f1_score(y_true, y_pred, average='micro')
            fw = f1_score(y_true, y_pred, average='weighted')
            if j != int(summaries):
                s += str(round(fma,4))+" & "
                s1 += str(round(fmi,3))+" & "
                s2 += str(round(fw,3))+" & "
            else:
                s +=  str(round(fma,4))+"\\\\ \n \\hline \n"
                s1 +=  str(round(fmi,3))+"\\\\ \n \\hline \n"
                s2 +=  str(round(fw,3))+"\\\\ \n \\hline \n"
            f.close()
    if summary == "3":
        summary = "2E"
    s += "\\end{tabular}} \n \caption{Macro-averaged F1 Score of "+label+".}\\label{"+model_name+summary+"F}\n"
    s += "\\end{table}\n\n"  
    s1 += "\\end{tabular}} \n \caption{Micro-averaged F1 Score of "+label+".}\\label{"+model_name+summary+"FM}\n"
    s1 += "\\end{table}\n\n" 
    s2 += "\\end{tabular}} \n \caption{Sample-weighted F1 Score  of "+label+".}\\label{"+model_name+summary+"FW}\n"
    s2 += "\\end{table}" 
    print(s)
    fs = open("F1/"+model_name+summary+"F1.txt","w+")
    fs.write(s) 
    fs.write(s2)   



if __name__ == "__main__":
    main()