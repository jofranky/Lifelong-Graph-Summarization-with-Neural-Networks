import argparse
import configparser
import site
import sys

site.addsitedir('../../lib')  # Always appends to end

print(sys.path)


from config_utils import config_util as cfg_u

from graph_summary_generator import summary as gsg

import pathlib
import os
import ast

import pickle
import numpy as np
import matplotlib.pyplot as plt
class pointsList():
    
    def __init__(self,points):
        self.points = points

     # utils
    def load(self, filename):
        """
        Load the given data file

        
        Args:
            filename (str): Filename
        """
        f = open(filename, 'rb')
        tmp_dict = pickle.load(f)
        f.close() 
        self.__dict__.update(tmp_dict) 
        
        
    def save(self, filename):
        """
        Save the class to a data file

        
        Args:
            filename (str): Filename
        """
        f = open(filename, 'wb')
        pickle.dump(self.__dict__, f, 2)
        f.close()  

class datapoints():

    def __init__(self, unchangedF1,unchangedP1,newF1,newP1,deletedF1,deletedP1,eqcs1N,avgEQC1,avgE1,jacV1,jacE1, kld1,
    unchangedF2,unchangedP2,newF2,newP2,deletedF2,deletedP2,eqcs2N,avgEQC2,avgE2,jacV2,jacE2, kld2,stdEQC1,stdE1,stdEQC2,stdE2):
        self.eqcs1N = eqcs1N
        self.avgEQC1 = avgEQC1
        self.avgE1 = avgE1
        self.jacV1 = jacV1
        self.jacE1 = jacE1
        self.kld1 = kld1
        
        self.unchangedF1 = unchangedF1
        self.unchangedP1 = unchangedP1 
        
        self.newF1 = newF1
        self.newP1 = newP1
        
        self.deletedF1= deletedF1
        self.deletedP1 = deletedP1
   
        
        #2AC
        self.eqcs2N = eqcs2N
        self.avgEQC2 = avgEQC2
        self.avgE2 = avgE2
        self.jacV2 = jacV2
        self.jacE2 = jacE2
        self.kld2 = kld2
        
        self.unchangedF2 = unchangedF2
        self.unchangedP2 = unchangedP2 
        
        self.newF2 = newF2 
        self.newP2 = newP2
        
        self.deletedF2= deletedF2
        self.deletedP2 = deletedP2
        
        
        #Added std
        self.stdEQC1 = stdEQC1
        self.stdE1 = stdE1
        self.stdEQC2 = stdEQC2
        self.stdE2 = stdE2


def main():
    plt.rcParams['font.size'] = 12
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str,dest='config', help='path to config file')  
    args = parser.parse_args()
    
    cfg = configparser.ConfigParser()
    cfg.read(args.config)
       
    filesN = cfg['GraphSummary']['filesN']
    
    points = []
    
    print(int(filesN))
    load_data = cfg.getboolean('WorkParameter','load_data')
    name = cfg['WorkParameter']['name']
     #file
    
    if not  os.path.exists(name):
        os.makedirs(name)
    
    y1= []  
    y2 = [] 
    for i in range(1,int(filesN)+1):
        base_dir = cfg_u.makePath(cfg['DataExchange']['basedir'+str(i)])
        file_name = str(cfg_u.makePath(cfg['GraphSummary']['save_file'+str(i)]))
        f = open(base_dir /"unary_measures"/"unary_measures.txt","r")

        lines = f.readlines() 
        #1AC
        eqcs1N = int(lines[1].split(" = ")[1].replace("\n",""))
        avgEQC1 = float(lines[2].split(" = ")[1].replace("\n",""))
        stdEQC1 = float(lines[3].split(" = ")[1].replace("\n",""))
        avgE1 = float(lines[4].split(" = ")[1].replace("\n",""))
        stdE1 = float(lines[5].split(" = ")[1].replace("\n",""))

         #2AC
        eqcs2N = int(lines[8].split(" = ")[1].replace("\n",""))
        avgEQC2 = float(lines[9].split(" = ")[1].replace("\n",""))
        stdEQC2 = float(lines[10].split(" = ")[1].replace("\n",""))
        avgE2 = float(lines[11].split(" = ")[1].replace("\n",""))
        stdE2 = float(lines[12].split(" = ")[1].replace("\n",""))


        eqcs1 = float(lines[15].split(" = ")[1].replace("\n",""))
        eqcs2 = float(lines[16].split(" = ")[1].replace("\n",""))
        y1.append(eqcs1)
        y2.append(eqcs2)
            
           
    file1 = []  

    for i in range(1, int(filesN)+1):
            file_name = str(cfg_u.makePath(cfg['GraphSummary']['save_file'+str(i)])).replace("graph_data_gs","")[5:]
            file1 .append(file_name)
    print("ok")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(file1, y1, linewidth=2.0,label = "1-Hop",color='b')
    ax.plot(file1, y2, linewidth=2.0,label = "2-Hop",color='r')
    plt.legend(loc="upper right")
    plt.ylabel('Number of unique EQCs')
    plt.xlabel('Snapshot')
    plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    plt.tick_params(axis='y', which='major')
    plt.tick_params(axis='x', which='major')
    
    
    plt.savefig(name+"/"+"AC_EQCsAll.pdf", bbox_inches = "tight")
    

    plt.close()
    
    
    




if __name__ == "__main__":
    main()
    
