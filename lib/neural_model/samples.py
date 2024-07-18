import torch

import torch_geometric

import pickle

class subgraphs():
    def __init__(self, hop,hashPairs,subgraphs,inputS,outputS):
        #hop of summary model 
        self.hop = hop
        
        #List of pairs of subject and respective hash 
        self.hashPairs = hashPairs
        #vertices name
        self.subgraphs = subgraphs
        
        #vertices subgraphs
        self.inputS = inputS
        self.outputS = outputS

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