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
        #self.num_features = len( self.feature_list_ )
        #for i,k in enumerate(self.k_folds):
         #   print("Fold ", i, "size",len(k)) 

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

    if  not load_data:
        for i in range(1,int(filesN)+1):
            base_dir = cfg_u.makePath(cfg['DataExchange']['basedir'+str(i)])
            file_name = str(cfg_u.makePath(cfg['GraphSummary']['save_file'+str(i)]))

            graph_dataC =  base_dir/ file_name
            gs = gsg.graph_for_summary()
            gs.load( graph_dataC )
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
        
            
            
            
            #1AC
            jacV1 = 0
            jacE1 = 0
            kld1 = 0
            entro1 = 0
            
            #2AC
            jacV2 = 0
            jacE2 = 0
            kld2 = 0
            entro2 = 0
            
        
            
            if i != 1:
                f = open(base_dir /"binary_measures"/"binary_measures.txt","r")
                lines = f.readlines() 
                #1AC
                jacV1 = float(lines[1].split(" = ")[1].replace("\n",""))
                kld1 = float(lines[2].split(" = ")[1].replace("\n",""))

                
                #2AC
                jacV2 = float(lines[5].split(" = ")[1].replace("\n",""))
                kld2 = float(lines[6].split(" = ")[1].replace("\n",""))
                
                

            #for added/deleted/unchanged
            
            #AC1
            prev1 = gs.prev_eqcs1
            first1 = gs.first_eqcs1
            current1 = gs.current_eqcs1
            
            unchangedF1 = len(current1.intersection(first1))
            unchangedP1 = len(current1.intersection(prev1))
            
            newF1 = len(current1 - first1)
            newP1 = len(current1 - prev1)
            
            deletedF1 = len(first1 - current1)
            deletedP1 = len(prev1 - current1)
            
            #AC2
            prev2 = gs.prev_eqcs2
            first2 = gs.first_eqcs2
            current2 = gs.current_eqcs2
            
            unchangedF2 = len(current2.intersection(first2))
            unchangedP2 = len(current2.intersection(prev2))
            
            newF2 = len(current2 - first2)
            newP2 = len(current2 - prev2)
            
            deletedF2= len(first2 - current2)
            deletedP2 = len(prev2 - current2)
            
           
            
                
               
                
            
                   
            points.append(
            datapoints(unchangedF1,unchangedP1,newF1,newP1,deletedF1,deletedP1,eqcs1N,avgEQC1,avgE1,jacV1,jacE1, kld1,
        unchangedF2,unchangedP2,newF2,newP2,deletedF2,deletedP2,eqcs2N,avgEQC2,avgE2,jacV2,jacE2, kld2,stdEQC1,stdE1,stdEQC2,stdE2))
        listP = pointsList(points)
        listP.savefig(name+"/points")
    else:
        listP = pointsList([])
        listP.load(name+"/points")
        points = listP.points
        
    file1 = []  
    file2 = []  
    for i in range(1, int(filesN)+1):
            file_name = str(cfg_u.makePath(cfg['GraphSummary']['save_file'+str(i)])).replace("graph_data_gs","")
            file1.append(file_name[5:])
            file2.append(file_name[5:])
    file2.pop(0)
    #1AC |EQCs|
    y1= []  
    y2 = []  
    for i in points:
        y1.append(i.eqcs1N )
        y2.append(i.eqcs2N )
    
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(file1, y1, linewidth=2.0,label = "1-Hop",color='b')
    ax.plot(file1, y2, linewidth=2.0,label = "2-Hop",color='r')
    plt.legend(loc="upper right")
    plt.ylabel('Number of EQCs')
    plt.xlabel('Snapshot')
    plt.tick_params(axis='y', which='major')
    plt.tick_params(axis='x', which='major')
    plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0)) 
    plt.savefig(name+"/"+"AC_EQCs.pdf", bbox_inches = "tight")
    #1AC avgEQC
    y = []  
    yd  = [] 
    for i in points:
        yd.append(i.stdEQC1  )
        y.append(i.avgEQC1 )
    y = np.array(y)
    yd = np.array(yd)
    
    
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(file1, y, 'b-',linewidth=2.0)
    ax.fill_between(file1, y- yd, y + yd, color='b', alpha=0.2)
    plt.ylabel('avgSize')
    plt.xlabel('Snapshot')
    plt.tick_params(axis='y', which='major')
    plt.tick_params(axis='x', which='major')
    plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0)) 
    plt.savefig(name+"/"+"1AC_avgEQC.pdf", bbox_inches = "tight")
    
    #1AC avgE 
    y = []  
    yd  = [] 
    for i in points:
        yd.append(i.stdE1  )
        y.append(i.avgE1 )
    y = np.array(y)
    yd = np.array(yd)
    
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(file1,y, 'b-',linewidth=2.0)
    ax.fill_between(file1, y- yd, y + yd, color='b', alpha=0.2)
    plt.ylabel('avgE')
    plt.xlabel('Snapshot')
    plt.tick_params(axis='y', which='major')
    plt.tick_params(axis='x', which='major')
    plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0)) 
    plt.savefig(name+"/"+"1AC_avgE.pdf", bbox_inches = "tight")
    
   
    
    #AC kld
    y1= []  
    y2 = []  
    for i in points:
        y1.append(i.kld1 )
        y2.append(i.kld2 )
    y1.pop(0)
    y2.pop(0)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(file2, y1, linewidth=2.0,label = "1-Hop",color='b')
    ax.plot(file2, y2, linewidth=2.0,label = "2-Hop",color='r')
    plt.legend(loc="upper right")
    plt.ylabel('JS')
    plt.xlabel('Snapshot')
    plt.tick_params(axis='y', which='major')
    plt.tick_params(axis='x', which='major')
    plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0)) 
    plt.savefig(name+"/"+"AC_kld.pdf", bbox_inches = "tight")
    
    
    #AC jacV
    y1= []  
    y2 = []  
    for i in points:
        y1.append(i.jacV1)
        y2.append(i.jacV2 )
    y1.pop(0)
    y2.pop(0)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(file2, y1, linewidth=2.0,label = "1-Hop",color='b')
    ax.plot(file2, y2, linewidth=2.0,label = "2-Hop",color='r')
    plt.legend(loc="upper right")
    plt.ylabel('jacV')
    plt.xlabel('Snapshot')
    plt.tick_params(axis='y', which='major')
    plt.tick_params(axis='x', which='major')
    plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0)) 
    plt.savefig(name+"/"+"AC_jacV.pdf", bbox_inches = "tight")
    
   
    
    
    plt.close()
    
    
    #1AC
    #unchanged,new,deleted
    yUnchangedF = []
    yUnchangedP = []
    yNewF = []
    yNewP= []
    yDeletedF = []
    yDeletedP = []
    

    for i in points:
        yUnchangedF.append(i.unchangedF1)
        yUnchangedP.append(i.unchangedP1)
        yNewF.append(i.newF1)
        yNewP.append(i.newP1)
        yDeletedF.append(i.deletedF1)
        yDeletedP.append(i.deletedP1)
    yUnchangedF.pop(0)
    yUnchangedP.pop(0)
    yNewF.pop(0)
    yNewP.pop(0)
    yDeletedF.pop(0)
    yDeletedP.pop(0)
    
    #unchanged
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(file2, yUnchangedP,"b", linewidth=2.0,label="Reoccurring")
    ax.plot(file2, yNewP,"b--", linewidth=2.0,label="Added")
    ax.plot(file2, yDeletedP,"b:", linewidth=2.0,label="Deleted")
    plt.legend(loc="upper right")
    plt.ylabel('Number of EQCs')
    plt.xlabel('Snapshot')
    plt.tick_params(axis='y', which='major')
    plt.tick_params(axis='x', which='major')
    plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0)) 
    plt.savefig(name+"/"+"1AC_changesPrev.pdf", bbox_inches = "tight")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(file2, yUnchangedF,"b",linewidth=2.0,label="Reoccurring")
    ax.plot(file2, yNewF,"b--",linewidth=2.0,label="Added")
    ax.plot(file2,yDeletedF,"b:",linewidth=2.0,label="Deleted")
    plt.legend(loc="upper right")
    plt.ylabel('Number of EQCs')
    plt.xlabel('Snapshot')
    plt.tick_params(axis='y', which='major')
    plt.tick_params(axis='x', which='major')
    plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0)) 
    plt.savefig(name+"/"+"1AC_changesFirst.pdf", bbox_inches = "tight")
    
    
    plt.close()
    
    
 

    #2AC avgEQC
    y = []  
    yd  = [] 
    for i in points:
        yd.append(i.stdEQC2  )
        y.append(i.avgEQC2)
    y = np.array(y)
    yd = np.array(yd)
    
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(file1,y, 'r-',linewidth=2.0)
    ax.fill_between(file1, y- yd, y + yd, color='r', alpha=0.2)
    plt.ylabel('avgEQC')
    plt.xlabel('Snapshot')
    plt.tick_params(axis='y', which='major')
    plt.tick_params(axis='x', which='major')
    plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0)) 
    plt.savefig(name+"/"+"2AC_avgEQC.pdf", bbox_inches = "tight")
    
    #2AC avgE 
    y = []  
    yd  = [] 
    for i in points:
        yd.append(i.stdE2  )
        y.append(i.avgE2)
    y = np.array(y)
    yd = np.array(yd)
    
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(file1,y, 'r-',linewidth=2.0)
    ax.fill_between(file1, y- yd, y + yd, color='r', alpha=0.2)
    plt.ylabel('avgE')
    plt.xlabel('Snapshot')
    plt.tick_params(axis='y', which='major')
    plt.tick_params(axis='x', which='major')
    plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0)) 
    plt.savefig(name+"/"+"2AC_avgE.pdf", bbox_inches = "tight")
    
    
    
    
    plt.close()
    #2AC
    #unchanged,new,deleted
    yUnchangedF = []
    yUnchangedP = []
    yNewF = []
    yNewP=  []
    yDeletedF =  []
    yDeletedP =  []
    
    for i in points:
        yUnchangedF.append(i.unchangedF2)
        yUnchangedP.append(i.unchangedP2)
        yNewF.append(i.newF2)
        yNewP.append(i.newP2)
        yDeletedF.append(i.deletedF2)
        yDeletedP.append(i.deletedP2)
    yUnchangedF.pop(0)
    yUnchangedP.pop(0)
    yNewF.pop(0)
    yNewP.pop(0)
    yDeletedF.pop(0)
    yDeletedP.pop(0)
    
     #unchanged
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(file2, yUnchangedP,"r", linewidth=2.0,label="Reoccurring")
    ax.plot(file2, yNewP,"r--", linewidth=2.0,label="Added")
    ax.plot(file2, yDeletedP,"r:", linewidth=2.0,label="Deleted")
    plt.legend(loc="upper right")
    plt.ylabel('Number of EQCs')
    plt.xlabel('Snapshot')
    plt.tick_params(axis='y', which='major')
    plt.tick_params(axis='x', which='major')
    plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0)) 
    plt.savefig(name+"/"+"2AC_changesPrev.pdf", bbox_inches = "tight")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(file2, yUnchangedF,"r",linewidth=2.0,label="Reoccurring")
    ax.plot(file2, yNewF,"r--",linewidth=2.0,label="Added")
    ax.plot(file2,yDeletedF,"r:",linewidth=2.0,label="Deleted")
    plt.legend(loc="upper right")
    plt.ylabel('Number of EQCs')
    plt.xlabel('Snapshot')
    plt.tick_params(axis='y', which='major')
    plt.tick_params(axis='x', which='major')
    plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0)) 
    plt.savefig(name+"/"+"2AC_changesFirst.pdf", bbox_inches = "tight")
    
    
    plt.close()
    
    



if __name__ == "__main__":
    plt.rcParams['font.size'] = 12
    main()
