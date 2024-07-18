import argparse
import configparser
import site
import sys
import pickle

site.addsitedir('../../lib')  # Always appends to end

print(sys.path)
import statistics

from config_utils import config_util as cfg_u

from graph_summary_generator import summary as gsg

import pathlib

import ast
import os

import numpy as np
import matplotlib.pyplot as plt

class graphInfo():
    def __init__(self,current_subjects_set,current_eqcs1,current_eqcs2,current_eqcs3,eqcs1,eqcs2,eqcs3,current_features,label_list1,label_list2  ):
        #Graph/Summary informations
        self.current_subjects_set =  current_subjects_set
        self.current_eqcs1 = current_eqcs1 
        self.current_eqcs2 = current_eqcs2
        self.current_eqcs3 = current_eqcs3
        self.eqcs1 = eqcs1
        self.eqcs2 = eqcs2
        self.eqcs3 = eqcs3
        self.current_features = current_features
        self.label_list1 = label_list1
        self.label_list2 = label_list2
        

class unary():
    
    def __init__(self,gs):
        
        #for calculating and saving measures
        self.gs = gs
        
        self.allMembers = 0
        
        #1-hop AC
        self.eqcs1 = 0
        self.avgEQC1 = 0 #avearge members
        self.stdEQC2 = 0 #std members
        self.sumEdges1 = 0
        self.avgE1 = 0 #Average edges
        self.stdE1 = 0 #stdfedges
        
        #2-hop AC
        self.eqcs2 = 0
        self.avgEQC2 = 0 #avearge members
        self.stdEQC2 = 0 #std members
        self.sumEdges2 = 0
        self.avgE2 = 0 #Average edges
        self.stdE2 = 0 #stdfedges

        
        #Histograms about EQCs degree
        #1-hop AC
        self.degrees1 = []
        self.maxD1 = 0
        #2-hop AC
        self.degrees2 = []
        self.maxD2 = 0
        
        #Histograms about members of EQCs
        #1-hop AC
        self.members1 = []
        self.maxM1 = 0
        #2-hop AC
        self.members2 = []
        self.maxM2 = 0

        #Histograms about the frequencey for properties
        self.freq1 = []
        self.maxF1 = 0
        self.eqcs1A = len(gs.label_list1)
        self.eqcs2A = len(gs.label_list2)

        
    
    def AC1Values(self,f):
        #graph 
        self.allMembers = len(self.gs.current_subjects_set)
        print("Original graph has "+str(self.allMembers)+" vertices")

        #AC1 
        self.eqcs1 = len(self.gs.current_eqcs1)
        print("1-hop Attribute Collection")
        print("|EQCs| = "+str(self.eqcs1))
        f.write("1-hop Attribute Collection\n")
        f.write("|EQCs| = "+str(self.eqcs1)+"\n")
        f.flush()

        self.avgEQC1 = self.allMembers/self.eqcs1
        print("AvgEQC = "+str(self.avgEQC1))
        f.write("AvgEQC = "+str(self.avgEQC1)+"\n")
        
        self.stdEQC1  = statistics.stdev(self.members1)
        print("StdEQC = "+str(self.stdEQC1))
        f.write("StdEQC = "+str(self.stdEQC1)+"\n")

        self.sumEdges1 = 0
        for e in self.gs.eqcs1.values():
             self.sumEdges1 += e.degree
        self.avgE1 = self.sumEdges1/self.eqcs1
        print("AvgE = "+str(self.avgE1))
        f.write("AvgE = "+str(self.avgE1)+"\n")
        self.stdE1  = statistics.stdev(self.degrees1)
        print("StdE = "+str(self.stdE1))
        f.write("StdE = "+str(self.stdE1)+"\n\n")
        f.flush()
        
    def AC2Values(self,f):
        #graph 
        self.allMembers = len(self.gs.current_subjects_set)
        print("Original graph has "+str(self.allMembers)+" vertices")

        #AC2
        self.eqcs2 = len(self.gs.current_eqcs2)
        print("2-hop Attribute Collection")
        print("|EQCs| = "+str(self.eqcs2))
        f.write("2-hop Attribute Collection\n")
        f.write("|EQCs| = "+str(self.eqcs2)+"\n")
        f.flush()

        self.avgEQC2 = self.allMembers/self.eqcs2
        print("AvgEQC = "+str(self.avgEQC2))
        f.write("AvgEQC = "+str(self.avgEQC2)+"\n")
        self.stdEQC2  = statistics.stdev(self.members2)
        print("StdEQC = "+str(self.stdEQC2))
        f.write("StdEQC = "+str(self.stdEQC2)+"\n")

        self.sumEdges2 = 0
        for e in self.gs.eqcs2.values():
             self.sumEdges2 += e.degree
        self.avgE2 = self.sumEdges2/self.eqcs2
        print("AvgE = "+str(self.avgE2))
        f.write("AvgE = "+str(self.avgE2)+"\n")
        self.stdE2 = statistics.stdev(self.degrees2)
        print("StdE = "+str(self.stdE2))
        f.write("StdE = "+str(self.stdE2)+"\n\n")
        f.flush()  
        
        
    def AC1Degrees(self):
        #Histograms about EQCs degree
        #AC1
        self.degrees1 = []
        self.maxD1 = 0
        for e in self.gs.eqcs1.values():
             self.degrees1.append(e.degree)
             if(e.degree > self.maxD1):
                  self.maxD1 = e.degree
        
        
    def AC1DegreesHist(self,base_dirC):
       #specify bin start and end points
        bin_ranges1 = list(range(self.maxD1+1))
        #create histogram with 4 bins
        plt.hist(self.degrees1, bins=bin_ranges1, edgecolor='black')
        # Save the histogram
        plt.ylabel('Number of EQCs')
        plt.xlabel('Number of Attributes')
        plt.yscale('log')
        plt.xscale('log')
        plt.savefig(base_dirC  /'degree1.pdf')
        plt.clf() 
    
    def AC2Degrees(self):
        #Histograms about EQCs degree
        #AC2
        self.degrees2 = []
        self.maxD2 = 0
        for e in self.gs.eqcs2.values():
             self.degrees2.append(e.degree)
             if(e.degree > self.maxD2):
                  self.maxD2 = e.degree
        
        
    def AC2DegreesHist(self,base_dirC):
       #specify bin start and end points
        bin_ranges2 = list(range(self.maxD2+1))
        #create histogram with 4 bins
        plt.hist(self.degrees2, bins=bin_ranges2, edgecolor='black')
        # Save the histogram
        plt.ylabel('Number of EQCs')
        plt.xlabel('Number of Attributes')
        plt.yscale('log')
        plt.xscale('log')
        plt.savefig(base_dirC  /'degree2.pdf')
        plt.clf() 
        
        
        

        
    def AC1Members(self):
        self.members1 = []
        self.maxM1 = 0
        for e in self.gs.eqcs1.values():
         self.members1.append(len(e.members))
         if(len(e.members) > self.maxM1):
           self.maxM1 = len(e.members) 
        
        
    def AC1MembersHist(self,base_dirC):
        bin_ranges1 = list(range(self.maxM1+1))
        plt.hist(self.members1, bins=bin_ranges1, edgecolor='black')
        plt.ylabel('Number of EQCs')
        plt.xlabel('Number of Members')
        plt.yscale('log')
        plt.xscale('log')
        plt.savefig(base_dirC /'member1.pdf')  
        plt.clf()
        
    def AC2Members(self):
        self.members2 = []
        self.maxM2 = 0
        for e in self.gs.eqcs2.values():
         self.members2.append(len(e.members))
         if(len(e.members) > self.maxM2):
           self.maxM2 = len(e.members) 
        
        
    def AC2MembersHist(self,base_dirC):
        bin_ranges2 = list(range(self.maxM2+1))
        plt.hist(self.members2, bins=bin_ranges2, edgecolor='black')
        plt.ylabel('Number of EQCs')
        plt.xlabel('Number of Members')
        plt.yscale('log')
        plt.xscale('log')
        plt.savefig(base_dirC /'member2.pdf')  
        plt.clf()

    
    
    def freqPre(self):
         #Histograms about the frequencey for properties and types
        self.freq1 = []
        self.maxF1 = 0
        for e in self.gs.current_features.values():
         self.freq1.append(e.frequency)
         if(e.frequency > self.maxF1):
           self.maxF1 = e.frequency 
        
        
        
    def freqPreHist(self,base_dirC): 
        bin_ranges1 = list(range(self.maxF1+1))
        plt.hist(self.freq1, bins=bin_ranges1, edgecolor='black')
        plt.xlabel('Number of Usages')
        plt.ylabel('Number of Predicates')
        plt.yscale('log')
        plt.xscale('log')
        plt.savefig(base_dirC /'freqAll.pdf')  
        plt.clf()
        
    
    
    
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
    
    def ACAll(self,f):

        print("All seen:")
        print("|EQCs1| = "+str(self.eqcs1A) )
        print("|EQCs2| = "+str(self.eqcs2A) )
        f.write("All seen: \n")
        f.write("|EQCs1| = "+str(self.eqcs1A)+"\n")
        f.write("|EQCs2| = "+str(self.eqcs2A)+"\n")
        f.flush()
     
#Calculating the unary measures from the appendix.
def main():
    
    #create graph Info
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str,dest='config', help='path to config file')  
    args = parser.parse_args()

    cfg = configparser.ConfigParser()
    cfg.read(args.config)
       
    basedirC = cfg_u.makePath(cfg['DataExchange']['basedirC'])
    load_data = cfg.getboolean('WorkParameter','load_data')
    plot =  cfg.getboolean('WorkParameter','plot')
    file_nameC = str(cfg_u.makePath(cfg['GraphSummary']['save_fileC']))
    print("Graphsummary file:", file_nameC)

    graph_dataC =  basedirC/ file_nameC
    
    g = gsg.graph_for_summary()
    g.load( graph_dataC )
    
    gs = graphInfo(g.current_subjects_set,g.current_eqcs1,g.current_eqcs2,g.current_eqcs3,g.eqcs1,g.eqcs2,g.eqcs3,g.current_features,g.label_list1,g.label_list2 )
    print("Graph is loaded")
    del g #free spaces
    
    
    #file
    if not  os.path.exists(basedirC / "unary_measures"):
        os.makedirs(basedirC / "unary_measures")
    name = basedirC / "unary_measures"/"unary_measures.txt"
    f = open(name,"w+")
    
    
    unaryMeasures = unary(gs)
    if not load_data:
    
        unaryMeasures.AC1Degrees()
        unaryMeasures.AC2Degrees()
        
        unaryMeasures.AC1Members()
        unaryMeasures.AC2Members()
        
        unaryMeasures.AC1Values(f)
        unaryMeasures.AC2Values(f)     
        
        unaryMeasures.freqPre()

        
        unaryMeasures.save(basedirC / "unary_measures"/"unary_measures")
    else:
        unaryMeasures.load(basedirC / "unary_measures"/"unary_measures")
        unaryMeasures.AC1Values(f)
        unaryMeasures.AC2Values(f)
    
    unaryMeasures.ACAll(f)
    #save histogramms
    if plot:
        base_dirC = basedirC / "unary_measures"
        unaryMeasures.AC1DegreesHist(base_dirC)
        unaryMeasures.AC2DegreesHist(base_dirC)
        unaryMeasures.AC1MembersHist(base_dirC)
        unaryMeasures.AC2MembersHist(base_dirC)
        unaryMeasures.freqPreHist(base_dirC)
   


if __name__ == "__main__":
    main()
