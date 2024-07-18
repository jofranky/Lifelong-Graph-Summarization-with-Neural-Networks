import argparse
import configparser
import site

site.addsitedir('../../lib')  # Always appends to end

import random

import torch
import torch_geometric
import torch.nn.functional as F
import timeit
from neural_model import mlp, saint,  utils, graphmlp


from config_utils import config_util as cfg_u
from neural_model import  samples  as sam
from torch.utils.tensorboard import SummaryWriter
    
from datetime import datetime
import gc
import ast
import os

def createSubgraphs(listSub,num_features,num_vertices ):
    x = torch.zeros( num_vertices , num_features ) 
    y = torch.zeros( num_vertices , dtype=torch.long ) 
    edge_index = torch.tensor( [ [], [] ], dtype=torch.long )
    offset = 0
    for s in listSub:
        for i,j in s['x']:
            x[i+offset,j] = 1
        y[offset] = s["y"]
        edge_indexA  = s["edge_index"]
        edge_indexA += offset
        edge_index = torch.cat( ( edge_index, edge_indexA), 1 )  
        edge_indexA -= offset
        offset  += s['vertices']
    data = torch_geometric.data.Data( x=x, y=y,edge_index=edge_index ) 
    return data

def draw_all( dataList ,subgraphs,num_features, guard_condition ):
    """
    Draw subgraph for validation or training
    """
    num_vertices = 0
    listSub  = []
    while( num_vertices  <  guard_condition and  dataList ):
    #for s, h in dataList:
        (s,h) = dataList.pop()
        sub = subgraphs[h]
        if sub["vertices"] + num_vertices > guard_condition:
            if sub["vertices"] < guard_condition:
                dataList.append((s,h))
            break
        listSub.append(sub)
        num_vertices += sub["vertices"] 
    data  = createSubgraphs( listSub,num_features,num_vertices )    
    return data

def evaluate_model( model, dataL, subgraphs,device, guard_condition,num_features,f): #accuracy for the whole data
    """
    Evaluate model using all data and batch size with  "guard_condition" samples

    """
    model.eval()
    tp  = 0
    n = guard_condition
    #calculate the average over multipe inferences
    nV = 0 #number of samples
    data_list = list(dataL)
    numberAll = len(dataL)
    log = 'Tested: {:.2f}%'
    while  data_list:      
        data = draw_all( data_list,subgraphs,num_features,guard_condition )        
        data = data.to(device)
        
        logits = model(data)   
        
        #filter out transductive and dummy vertices
        mask = [data.y > 0]   
        
        tp += calc_tp( logits[mask], data.y[mask] ,f)
        nV += data.y[mask].size()[0]
        numberRest = len(data_list)
        ready =  ((numberAll- numberRest)/numberAll)*100
        print(log.format(ready))
        gc.collect()
        torch.cuda.empty_cache() #free memory after training/testing
    return ( tp  / nV ) 
    
def calc_tp(logits, y,f ):
    #get prediction
    pred = logits.max(1)[1]
    f.write("Predicted :"+ str(pred)+"\n") # saving the predictions of the model and the real values
    f.write("Real :"+str(y)+"\n\n")
    return pred.eq(y).sum().item()

def calc_acc( logits, y ):s
    """
    Calculates the needed data to train the networks

    
    Args:
        logits (list): Predicted class labels
        y (list): Target class labels
    """
    #get prediction
    pred = logits.max(1)[1]
    return pred.eq(y).sum().item() / y.size()[0]

   


def main():    
    random.seed(42)
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str,dest='config', help='path to config file')  
    args = parser.parse_args()
        
    cfg = configparser.ConfigParser()
    print(args)
    cfg.read(args.config)
    
    epochs = cfg.getint('WorkParameter', 'epochs')
    guard = cfg.getint('WorkParameter', 'guard')  
    
    cuda_core = cfg.get('WorkParameter', 'cuda_core')    
   
    basedirT= cfg_u.makePath(cfg['DataExchange']['basedirTest'])
    ac1Path = basedirT / "1AC"
    ac2Path = basedirT / "2AC"
    ac3Path = basedirT / "2ACE"
    
    test =  float(cfg['WorkParameter']['test'])
    val =   float(cfg['WorkParameter']['validation'])
    train=  float(cfg['WorkParameter']['training'])
    if test+val +train != 1.0:
        print("Error")
    
    
    sub = sam.subgraphs(0,[],{},0,0)
    summary = cfg.get('WorkParameter', 'summary')
    #load subgraphs for training according to summary model
    if summary == "1":
        sub.load( ac1Path )
    elif summary == "2":
        sub.load( ac2Path )
    elif summary == "3":
        sub.load( ac3Path )
    else:
        print("Unknown summary model: ",summary)
    
     
    lenPairs = len(sub.hashPairs)
    testS = int( test*lenPairs)
    valS = int(val*lenPairs)
    testSet = sub.hashPairs[0:testS]
    valSet = sub.hashPairs[testS:(testS+valS)]
    trainSet = sub.hashPairs[(testS+valS):]
    subgraphs = sub.subgraphs
    
    weightsTrain = {}
    for s,h in trainSet:
        if h in weightsTrain:
            weightsTrain[h] += 1
        else:
            weightsTrain[h]  = 1
            
    weights = []
    trainLen = len(trainSet)
    for i in range(trainLen):
        s,h = trainSet[i]
        weights.append(1-(weightsTrain[h] /trainLen))

    model_name =  cfg.get('GNN', 'model_name')
    learning_rate =  cfg.getfloat('GNN', 'learning_rate')
    weight_decay =  cfg.getfloat('GNN', 'weight_decay')
    dropout =  cfg.getfloat('GNN', 'dropout')
    hidden_layer =  cfg.getint('GNN', 'hidden_layer')    
    num_features =  sub.inputS
    num_classes =  sub.outputS+1 # because we have a 0 class that we ignore
    tau =  cfg.getfloat('GNN', 'tau')
    alpha =  cfg.getfloat('GNN', 'alpha')
    k_hop =  cfg.getfloat('GNN', 'k_hop')    
    h_layers =  cfg.getint('GNN', 'h_layers')
    
    basedirP = cfg_u.makePath(cfg['DataExchange']['basedir'])
    s = model_name+"_"+summary
   
    s1= str(basedirT).split("/")[-1]
    if not  os.path.exists(basedirP / "Tests"):
            os.makedirs(basedirP / "Tests")
    s2 =  s+"_"+s1+".txt"
    check =  basedirP / "Tests"/ s2
    f = open(check, "w")
    
        
    device = torch.device(cuda_core if torch.cuda.is_available() else 'cpu')
    print("Running on ", device)
    pathP = str( basedirP / "TrainedModels"/ s)
    parametersP = torch.load(pathP)
    prevInput  = 0
    prevOutput = 0
    if ( model_name == "mlp" ):        
        prevInput = len(parametersP["model_state_dict"]["lin1.weight"][0])
        prevOutput = len(parametersP["model_state_dict"]["lin2.weight"])
        if prevOutput > num_classes:
            num_classes=prevOutput
        if prevInput > num_features:
            num_features = prevInput
        model = mlp.MLP( num_features, num_classes, hidden_layer, dropout = dropout )
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif model_name == "graphsaint":
        prevInput = len(parametersP["model_state_dict"]["conv1.lin_rel.weight"][0])
        prevOutput = len(parametersP["model_state_dict"]["lin.weight"])
        if prevOutput > num_classes:
            num_classes=prevOutput
        if prevInput > num_features:
            num_features = prevInput
        model = saint.SAINT( num_features, num_classes, hidden_layer, dropout = dropout,h_layers=h_layers )
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif model_name == "graphmlp":
        prevInput = len(parametersP["model_state_dict"]["mlp.fc1.weight"][0])
        prevOutput = len(parametersP["model_state_dict"]["classifier.weight"])
        if prevOutput > num_classes:
            num_classes=prevOutput
        if prevInput > num_features:
            num_features = prevInput
        model = graphmlp.GMLP( num_features, num_classes, hidden_layer, dropout = dropout, device = device )
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    
    
    if model_name == "graphsaint":
        prevInput = len(parametersP["model_state_dict"]["conv1.lin_rel.weight"][0])

        prevOutput = len(parametersP["model_state_dict"]["lin.weight"])

        if prevInput < num_features:
            f1zeros = torch.zeros( hidden_layer, num_features-prevInput).to(device) 
            parametersP["model_state_dict"]["conv1.lin_rel.weight"]  = torch.cat( ( parametersP["model_state_dict"]["conv1.lin_rel.weight"], f1zeros ) , 1 )  
            f2zeros = torch.zeros( hidden_layer, num_features-prevInput).to(device) 
            parametersP["model_state_dict"]["conv1.lin_root.weight"]  = torch.cat( ( parametersP["model_state_dict"]["conv1.lin_root.weight"], f2zeros) , 1 )  
        if prevOutput < num_classes:
            c1zeros =  torch.zeros( num_classes - prevOutput, (1+h_layers)*hidden_layer).to(device)
            parametersP["model_state_dict"]["lin.weight"] = torch.cat( ( parametersP["model_state_dict"]["lin.weight"],c1zeros) , 0 )  
            c2zeros = torch.zeros( num_classes - prevOutput ).to(device)
            parametersP["model_state_dict"]["lin.bias"] = torch.cat( ( parametersP["model_state_dict"]["lin.bias"],c2zeros ), 0 )  

        model.load_state_dict(parametersP['model_state_dict'])
        model.eval()   
    elif model_name == "mlp":
        prevInput = len(parametersP["model_state_dict"]["lin1.weight"][0])

        prevOutput = len(parametersP["model_state_dict"]["lin2.weight"])

        if prevInput < num_features:
            fzeros = torch.zeros( hidden_layer, num_features-prevInput).to(device) 
            parametersP["model_state_dict"]["lin1.weight"]  = torch.cat( ( parametersP["model_state_dict"]["lin1.weight"], fzeros) , 1 )            
        if prevOutput < num_classes:
            c1zeros = torch.zeros( num_classes - prevOutput, hidden_layer).to(device) 
            parametersP["model_state_dict"]["lin2.weight"] = torch.cat( ( parametersP["model_state_dict"]["lin2.weight"],c1zeros ) , 0 )  
            c2zeros = torch.zeros( num_classes - prevOutput ).to(device) 
            parametersP["model_state_dict"]["lin2.bias"] = torch.cat( ( parametersP["model_state_dict"]["lin2.bias"], c2zeros), 0 )  

        model.load_state_dict(parametersP['model_state_dict'])
        model.eval()   
    elif model_name == "graphmlp":
        prevInput = len(parametersP["model_state_dict"]["mlp.fc1.weight"][0])

        prevOutput = len(parametersP["model_state_dict"]["classifier.weight"])
        print("weight1Len ",len(parametersP["model_state_dict"]["classifier.weight"]))
        if prevInput < num_features:
            fzeros = torch.zeros( hidden_layer, num_features-prevInput).to(device)    
            parametersP["model_state_dict"]["mlp.fc1.weight"]  = torch.cat( ( parametersP["model_state_dict"]["mlp.fc1.weight"], fzeros) , 1 )            
        if prevOutput < num_classes:
            c1zeros =  torch.zeros( num_classes - prevOutput, hidden_layer).to(device) 
            parametersP["model_state_dict"]["classifier.weight"] = torch.cat( ( parametersP["model_state_dict"]["classifier.weight"],c1zeros) , 0 )  
            c2zeros = torch.zeros( num_classes - prevOutput ).to(device) 
            parametersP["model_state_dict"]["classifier.bias"] = torch.cat( ( parametersP["model_state_dict"]["classifier.bias"], c2zeros), 0 )  

        model.load_state_dict(parametersP['model_state_dict'])
        model.eval()   
        print("weight1Len ",len(parametersP["model_state_dict"]["classifier.weight"]))

    print("Model:",model)
    print("Parameter count:", sum(p.numel() for p in model.parameters()))        
    print("Optimizer:",optimizer)
    print(torch.cuda.is_available())
    model = model.to(device)      
    
    val_step = cfg.getint('WorkParameter', 'val_step')
    

    
    start_train = timeit.default_timer()

    test_acc = evaluate_model( model, testSet,subgraphs, device, guard,num_features,f)                            
    log = ' Test: {:.4f}'
    print(log.format( test_acc))
    f.write(log.format( test_acc))
    f.close()   


if __name__ == "__main__":
    main()

