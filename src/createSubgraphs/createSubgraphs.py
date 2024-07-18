import argparse
import configparser
import site
import sys
import random
from datetime import datetime
import torch
import torch_geometric

site.addsitedir('../../lib')  # Always appends to end

print(sys.path)


from config_utils import config_util as cfg_u

from graph_summary_generator import  summary  as gsg
from neural_model import  samples  as sam
import pathlib




def getNHopEdge(subgraph,num_features,  feature_index, name,hops, subjects, number):
    #print("Hello ", hops)
    s_index = number
    vertex = subjects[name]
    if hops != 1:
        for e in vertex.edges:
                # set id for feature matrix
                p_index = feature_index[e[0]]
                subgraph["x"].add(( s_index, p_index ))
                if e[1] in subjects:
                    number += 1
                    subgraph["x"].add((number, p_index ))
                    subgraph["edge_index"] = torch.cat( (  subgraph["edge_index"], torch.tensor([ [s_index], [number] ]) ), 1 )
                    number += 1 # number is now the object
                    getNHopEdge(subgraph,num_features,  feature_index,e[1],hops-1,subjects,number)
                    subgraph["edge_index"] = torch.cat( (  subgraph["edge_index"], torch.tensor([ [number-1], [number] ]) ), 1 )
    elif hops == 1:
         for e in vertex.edges:
                # set id for feature matrix
                p_index = feature_index[e[0]]
                subgraph["x"].add( (s_index, p_index ))  #adapt label for subjects
                
                
def createSubgraphEdge(gt,vertexName,hops):
    subjects = gt.current_subjects
    num_features = gt.sizeFeature
    
    feature_index = gt.feature_index
    
    subgraph = {}
    subgraph["y"] = 0
    subgraph["x"] = set()
    subgraph["edge_index"] = torch.tensor( [ [], [] ], dtype=torch.long )
    
    getNHopEdge(subgraph,num_features,  feature_index, vertexName,hops,subjects,0)
    if hops == 2:
        subgraph["vertices"] =  2*len(subjects[vertexName].edges)+1
        subgraph["y"]  = subjects[vertexName].eqcs2Index + 1 #+1 because 0  indicated i don't care for loss
    else:
        print("No summary model with ",hops," ! Every label is 0!")
    return subgraph
 
def getNHop(subgraph,num_features,  feature_index, name,hops, subjects, number):
    #print("Hello ", hops)
    s_index = number
    vertex = subjects[name]
    if hops != 1:
        for e in vertex.edges:
                # set id for feature matrix
                p_index = feature_index[e[0]]
                subgraph["x"].add(( s_index, p_index ))
                if e[1] in subjects:
                    number += 1
                    getNHop(subgraph,num_features,  feature_index,e[1],hops-1,subjects,number)
                    subgraph["edge_index"]  = torch.cat( ( subgraph["edge_index"], torch.tensor([ [s_index], [number] ]) ), 1 )
    elif hops == 1:
         for e in vertex.edges:
                # set id for feature matrix
                p_index = feature_index[e[0]]
                subgraph["x"].add(( s_index, p_index ))#adapt label for subjects
    
def createSubgraph(gt,vertexName,hops):
    subjects = gt.current_subjects
    num_features = gt.sizeFeature
    
    feature_index = gt.feature_index
    
    subgraph = {}
    subgraph["y"] = 0
    subgraph["x"] = set()
    subgraph["edge_index"] = torch.tensor( [ [], [] ], dtype=torch.long )
    if hops == 1:
        subgraph["vertices"] =  1
    else :
        subgraph["vertices"] =  len(subjects[vertexName].edges)+1
    # create needed data structures
    getNHop(subgraph,num_features,  feature_index, vertexName,hops,subjects,0)
    if  hops == 1:
        subgraph["y"]  = subjects[vertexName].eqcs1Index + 1  #+1 because 0  indicated i don't care for loss
    elif hops == 2:
       subgraph["y"]  = subjects[vertexName].eqcs2Index + 1  #+1 because 0  indicated i don't care for loss
    else:
        print("No summary model with ",hops," ! Every label is 0!")
    return subgraph
 
    
def main():    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str,dest='config', help='path to config file')  
    args = parser.parse_args()

    cfg = configparser.ConfigParser()
    cfg.read(args.config)
    
    base_dirC = cfg_u.makePath(cfg['DataExchange']['basedirC'])
  
    file_nameT = str(cfg_u.makePath(cfg['GraphSummary']['save_fileT']))
    graph_dataT =  base_dirC / file_nameT
    
    maxDegree = cfg.getint('WorkParameter', 'maxDegree')  
    
    gt =  gsg.t_graph({},{},[],[],[],[])
    gt.load( graph_dataT)
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    f = open(str(current_time)+".txt" , "w")
    f.write(file_nameT+"\n")
    f.write("All vertices: "+str(len(gt.current_subjects ))+"\n")
    print("All vertices: ",len(gt.current_subjects ))
    subjects = gt.current_subjects
    eqcs1Ex = {}
    eqcs2Ex = {}
    eqcs1Hash = []
    eqcs2Hash = []
    #find all possible subrgraphs of 1AC
    for s in subjects.keys():
        hash1 = subjects[s].hash1
        if hash1  not  in eqcs1Ex :
            eqcs1Ex[hash1] = s
        eqcs1Hash.append((s,hash1))
     #find all possible subrgraphs of 2AC  
    for s in subjects.keys():
        hash2 = subjects[s].hash2
        if hash2  not  in eqcs2Ex:
            eqcs2Ex[hash2] = s
        eqcs2Hash.append((s,hash2))
    
    print("EQCs1: ",len(eqcs1Ex))
    print("EQCs2: ",len(eqcs2Ex))
    f.write("EQCs1: "+str(len(eqcs1Ex))+"\n")
    f.write("EQCs2: "+str(len(eqcs2Ex))+"\n")
    eqcs1Sub = {}
    eqcs2Sub = {}
    eqcs2SubE = {}
    #print(eqcs1Hash,"\n\n")

    random.Random(42).shuffle(eqcs1Hash)
    
    for h in eqcs1Ex.keys():
        s = eqcs1Ex[h]
        eqcs1Sub[h]  = createSubgraph(gt,s,1)
        
    sub1 = sam.subgraphs(1,eqcs1Hash ,eqcs1Sub,gt.sizeFeature,gt.size1)
    save1 =  base_dirC / "1AC"
    sub1.save(save1)
    print("EQCs1Limited: ",len(eqcs1Sub))
    f.write("EQCs1Limited: "+str(len(eqcs1Sub))+"\n")
    del sub1
    del eqcs1Sub
    
    eqcs2ExL = {} # remove all EQCs of 2AC of vertices with a degree higher than maxDegree
    for h in eqcs2Ex.keys():
        s = eqcs2Ex[h]
        if len(gt.current_subjects[s].edges) <= maxDegree:  
            eqcs2Sub[h]  = createSubgraph(gt,s,2)
            eqcs2ExL[h] = s 
            
    eqcs2LHash= [] # removed all pairs of  EQCs of 2AC of vertices with a degree higher than maxDegree
    for s in subjects.keys():
        hash2 = subjects[s].hash2
        if hash2    in eqcs2ExL :
            eqcs2ExL[hash2] = s
            eqcs2LHash.append((s,hash2))
        
    random.Random(42).shuffle(eqcs2LHash)
        
    sub2 = sam.subgraphs(2,eqcs2LHash ,eqcs2Sub,gt.sizeFeature,gt.size2)
    save2 =  base_dirC / "2AC"
    sub2.save(save2)
    print("EQCs2Limited: ",len(eqcs2Sub))
    f.write("EQCs2Limited: "+str(len(eqcs2Sub))+"\n")
    del sub2
    del eqcs2Sub

    eqcs2ExL = {} # remove all EQCs of 2AC of vertices with a degree higher than maxDegree
    for h in eqcs2Ex.keys():
        s = eqcs2Ex[h]
        if len(gt.current_subjects[s].edges) <= maxDegree:  
            eqcs2SubE[h]  = createSubgraphEdge(gt,s,2)
            eqcs2ExL[h] = s 
            
    eqcs2LHash= [] # removed all pairs of  EQCs of 2AC of vertices with a degree higher than maxDegree
    for s in subjects.keys():
        hash2 = subjects[s].hash2
        if hash2    in eqcs2ExL :
            eqcs2ExL[hash2] = s
            eqcs2LHash.append((s,hash2))
        
    random.Random(42).shuffle(eqcs2LHash)
     
    sub2E = sam.subgraphs(3,eqcs2LHash ,eqcs2SubE,gt.sizeFeature,gt.size2)
    save2E =  base_dirC / "2ACE"
    sub2E.save(save2E)
    print("All vertices of A1: ",len(eqcs1Hash ))
    print("All vertices of A2: ",len(eqcs2LHash ))
    f.write("All vertices of A1: "+str(len(eqcs1Hash ))+"\n")
    f.write("All vertices of A2: "+str(len(eqcs2LHash ))+"\n")


    
if __name__ == "__main__":
    main()
