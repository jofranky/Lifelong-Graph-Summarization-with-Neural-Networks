import argparse
import configparser
import site

site.addsitedir('../../lib')  # Always appends to end
import gc
import random

import torch
import torch_geometric
import torch.nn.functional as F
import timeit
from neural_model import mlp, saint,  utils, graphmlp, saintN


from config_utils import config_util as cfg_u
from neural_model import  samples  as sam
from torch.utils.tensorboard import SummaryWriter
    
from datetime import datetime

import ast
import os


def calc_acc( logits, y ):
    """
    Calculates the needed data to train the networks

    
    Args:
        logits (list): Predicted class labels
        y (list): Target class labels
    """
    #get prediction
    pred = logits.max(1)[1]
    return pred.eq(y).sum().item() / y.size()[0]
    
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
    
def draw_samples( dataList,weights,subgraphs, guard_condition,num_features,trainSamples,f  ):
    """Importent for me!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    Draw subgraph samples for mini batch

    
    Args:
        guard_condition (int): Mini batch size
    """
    samples = 0
    num_vertices = 0
    listSub  = []
    while( num_vertices  <  guard_condition ):
        s,h= random.choices( population=dataList, weights=weights)[0]
        sub = subgraphs[h]
        if sub["vertices"] + num_vertices > guard_condition:
            break
        listSub.append(sub)
        num_vertices += sub["vertices"] 
        samples += 1
    data  = createSubgraphs( listSub,num_features,num_vertices )    
    trainSamples += samples
    f.write("Trained on "+str(trainSamples)+" samples until now\n")
    return trainSamples,data


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
   
    basedirC = cfg_u.makePath(cfg['DataExchange']['basedirC'])
    ac1Path = basedirC / "1AC"
    ac2Path = basedirC / "2AC"
    ac3Path = basedirC / "2ACE"
    
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
    
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    s1  = ""
    if ( model_name == "mlp" ):  
        s1 =  current_time + "_" + str(cfg_u.makePath("su"+summary+  "_mn" + model_name + "_lr"   + str(learning_rate).replace(".", "-")+ "_hl" + str(hidden_layer) + "_dr" + str(dropout).replace(".", "-")+"_wd" + str(weight_decay).replace(".", "-")))

    elif model_name == "graphsaint":
        s1 =  current_time + "_" + str(cfg_u.makePath("su"+summary+  "_mn" + model_name + "_lr"   + str(learning_rate).replace(".", "-")+ "_hl" + str(hidden_layer) + "_dr" + str(dropout).replace(".", "-")+"_hln"+str(h_layers )+"_wd" + str(weight_decay).replace(".", "-")+"_hln"+str(h_layers )))
    
    elif model_name == "graphmlp":
        s1 =  current_time + "_" + str(cfg_u.makePath("su"+summary+  "_mn" + model_name + "_lr"   + str(learning_rate).replace(".", "-")+ "_hl" + str(hidden_layer) + "_dr" + str(dropout).replace(".", "-")+"_hln"+str(h_layers )+ "_wd" + str(weight_decay).replace(".", "-")+"_tau" + str(tau).replace(".", "-") + "_alpha" + str(alpha).replace(".", "-")+"_ khop"+str(k_hop) ))
    
    s = model_name+"_"+summary
    if not  os.path.exists(basedirC / "TrainedModels"):
            os.makedirs(basedirC / "TrainedModels")
    check =  basedirC / "TrainedModels"/ s
    checkpoint_file =  str( check )
   
    writer_dir = basedirC / cfg_u.makePath("train_"+model_name) / s1
    writer = SummaryWriter( str(writer_dir) ) 

    s2 = s1+".txt"
    check2 =  basedirC / "TrainedModels"/ s2
    f = open(check2, "w")
    trainSamples  = 0
    f.write("args "+str(args) +"\n")
    
        
    device = torch.device(cuda_core if torch.cuda.is_available() else 'cpu')
    print("Running on ", device)

    if ( model_name == "mlp" ):        
        model = mlp.MLP( num_features, num_classes, hidden_layer, dropout = dropout )
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif model_name == "graphsaint":
        model = saint.SAINT( num_features, num_classes, hidden_layer, dropout = dropout,h_layers=h_layers )
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif model_name == "graphsaintN":
        model = saintN.SAINTN( num_features, num_classes, hidden_layer, dropout = dropout,h_layers=h_layers )
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif model_name == "graphmlp":
        model = graphmlp.GMLP( num_features, num_classes, hidden_layer, dropout = dropout, device = device )
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    loadM  = cfg.getboolean('WorkParameter','loadM')
    #for loading the weights of previous model depending on model type
    if loadM:
        basedirP = cfg_u.makePath(cfg['DataExchange']['basedirP'])
        pathP = str( basedirP / "TrainedModels"/ s)
        parametersP = torch.load(pathP)
        
        if model_name == "graphsaint"  :
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
        elif  model_name == "graphsaintN" :
            prevInput = len(parametersP["model_state_dict"]["conv1.lin.weight"][0])
          
            prevOutput = len(parametersP["model_state_dict"]["lin.weight"])
            
            if prevInput < num_features:
                f1zeros = torch.zeros( hidden_layer, num_features-prevInput).to(device) 
                parametersP["model_state_dict"]["conv1.lin.weight"]  = torch.cat( ( parametersP["model_state_dict"]["conv1.lin.weight"], f1zeros ) , 1 )   
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
    
    f.write("Model:"+str(model)+"\n")
    f.write("Parameter count:"+ str(sum(p.numel() for p in model.parameters()))+"\n")     
    f.write("Optimizer:"+str(optimizer)+"\n")
    f.write(str(torch.cuda.is_available())+"\n")
    f.write("All possible samples: "+str(len(sub.hashPairs))+"\n")
    
    
    val_step = cfg.getint('WorkParameter', 'val_step')
    

    
    start_train = timeit.default_timer()
    ##training of MLP or GraphSaint
    if model_name != "graphmlp":
        for epoch in range(epochs): 
            print(epoch)
            f.write(str(epoch)+"\n")
            start_ep = timeit.default_timer()
            trainSamples,data = draw_samples(trainSet,weights,subgraphs,guard,num_features,trainSamples,f ) 
            data = data.to(device)
           
            model.train()
            optimizer.zero_grad()
            out = model(data)
            
            #filter out transductive and dummy vertices
            mask = [data.y > 0]         #all vertices for loss that means no dummy or object vertices
            loss = F.nll_loss(out[mask], data.y[mask])
            
            loss.backward()
            optimizer.step()

            train_acc = calc_acc( out, data.y )        
            writer.add_scalar('Loss/train', loss, epoch)
            writer.add_scalar('Accuracy/train', train_acc, epoch)
            
            del data
            
            log = 'Epoch: {:03d}, Train: {:.4f}'
            print(log.format(epoch, train_acc))
            f.write(log.format(epoch, train_acc)+"\n")
            f.flush()
           

            writer.add_scalars('Accuracy', {"train": train_acc }, epoch)          
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss
                }, checkpoint_file )
           
            end_ep = timeit.default_timer()
            overall_timediff = end_ep - start_train
            print('Epoch time: {:.02f}; Time overall {:.02f}; ETA {:.02f}'.
                      format(end_ep - start_ep, overall_timediff
                             , ( overall_timediff / ( epoch + 1 ) ) * ( len(testSet) - ( epoch + 1 ) ) ) )
            print("Before: ",torch.cuda.memory_reserved(0))
            print("Before: ",torch.cuda.memory_allocated(0))
            
            torch.cuda.empty_cache() #free memory after training/testing
            gc.collect()
            print("After: ",torch.cuda.memory_reserved(0))
            print("After: ",torch.cuda.memory_allocated(0))
    else:
        for epoch in range(epochs):  
            print(epoch)
            f.write(str(epoch)+"\n")
            start_ep = timeit.default_timer()
            trainSamples,data = draw_samples(trainSet,weights,subgraphs,guard,num_features,trainSamples,f ) 
            adj_label = utils.get_A_r(data, k_hop)
            adj_label = adj_label.to(device)
            
            data = data.to(device)
            
            model.train()
            optimizer.zero_grad()
            output, x_dis = model(data)
            
            mask = [data.y > 0]        
            
            loss_train_class = F.nll_loss(output[mask], data.y[mask])
            loss_Ncontrast = utils.Ncontrast(x_dis, adj_label, tau = tau)
            loss_train = loss_train_class + loss_Ncontrast * alpha
            loss_train.backward()
            optimizer.step()
            
            train_acc = calc_acc(output[mask], data.y[mask])
            writer.add_scalar('Loss/train', loss_train, epoch)
            writer.add_scalar('Accuracy/train', train_acc, epoch)
            writer.add_scalar('Loss/normal', loss_train_class, epoch)
            writer.add_scalar('Loss/ncontrast', loss_Ncontrast, epoch)
            
            del data     
            
            log = 'Epoch: {:03d}, Train: {:.4f}'
            print(log.format(epoch, train_acc))
            f.write(log.format(epoch, train_acc)+"\n")
                
            writer.add_scalars('Accuracy', {"train": train_acc }, epoch)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss_train
                }, checkpoint_file )
            
            end_ep = timeit.default_timer()
            overall_timediff = end_ep - start_train
            print('Epoch time: {:.02f}; Time overall {:.02f}; ETA {:.02f}'.
                      format(end_ep - start_ep, overall_timediff
                             , ( overall_timediff / ( epoch + 1 ) ) * ( len(testSet) - ( epoch + 1 ) ) ) )
            print("Before: ",torch.cuda.memory_reserved(0))
            print("Before: ",torch.cuda.memory_allocated(0))
        
            torch.cuda.empty_cache() #free memory after training/testing
            gc.collect()
            print("After: ",torch.cuda.memory_reserved(0))
            print("After: ",torch.cuda.memory_allocated(0))

if __name__ == "__main__":
    main()

