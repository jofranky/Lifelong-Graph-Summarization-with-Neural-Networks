import argparse
import configparser
import site
import sys

site.addsitedir('../../lib')  # Always appends to end

print(sys.path)


from config_utils import config_util as cfg_u

from graph_summary_generator import summary as gsg

import pathlib

import ast



#disable randomization of hash
import os
import sys
hashseed = os.getenv('PYTHONHASHSEED')
if not hashseed:
  os.environ['PYTHONHASHSEED'] = '0'
  os.execv(sys.executable, [sys.executable] + sys.argv)


#Create the 1-AC and 2-AC of a snapshot. Additionally, information is saved.
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str,dest='config', help='path to config file')  
    args = parser.parse_args()

    cfg = configparser.ConfigParser()
    cfg.read(args.config)
    
    load_data = cfg.getboolean('WorkParameter', 'load_data')
        
    base_dirC = cfg_u.makePath(cfg['DataExchange']['basedirC'])
   
    filtered_datafileC  =  base_dirC / cfg_u.makePath(cfg['Dyldo']['filtered_datafileC'])

    
    file_nameC = str(cfg_u.makePath(cfg['GraphSummary']['save_fileC']))
    file_nameT = str(cfg_u.makePath(cfg['GraphSummary']['save_fileT']))
    print("Graphsummary file:", file_nameC)
    graph_dataC =  base_dirC / file_nameC
    graph_dataT =  base_dirC / file_nameT
    prev = cfg.getboolean('GraphSummary','Prev')
    
    if load_data:  
        gs = gsg.graph_for_summary( )
        gs.load( graph_dataC )
        print("Graph is loaded")
    elif prev: 
        base_dirP = cfg_u.makePath(cfg['DataExchange']['basedirP'])
        file_nameP = str(cfg_u.makePath(cfg['GraphSummary']['save_fileP']))
        graph_dataP =  base_dirP  / file_nameP
        prevG = gsg.graph_for_summary( )
        prevG.load( graph_dataP) 
        
        gs = gsg.graph_for_summary(prev = prevG)
        gs.create_graph_information( filtered_datafileC )
        gs.calculate_graph_summary( )
        gs.save( graph_dataC ) 
        gs.createT(graph_dataT )
    else:
        gs = gsg.graph_for_summary( )
        print("Path"+str(filtered_datafileC ))
        gs.create_graph_information( filtered_datafileC )
        gs.calculate_graph_summary( )
        gs.save( graph_dataC ) 
        gs.createT(graph_dataT )
        
        
    print("Model: [","num_vertices:",gs.num_vertices,"num_features:", gs.get_num_features(), "num_classes1:",gs.get_num_classes1(), "num_classes2:",gs.get_num_classes2(),"]")

    
    
    
if __name__ == "__main__":
    main()
