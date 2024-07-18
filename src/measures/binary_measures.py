import argparse
import configparser
import site
import sys
import math
site.addsitedir('../../lib')  # Always appends to end

print(sys.path)
import os

from config_utils import config_util as cfg_u

from graph_summary_generator import summary as gsg

#Calculating the binary measures from the appendix.

def KLD(num_verticesC,eqcsC, num_verticesP,eqcsP):
    d_t_t1 = 0
    for hash,eq in eqcsC.items():
        p_t1 = len(eq.members)/num_verticesC
        if hash in eqcsP.keys():
            p_t = len(eqcsP[hash].members)/num_verticesP
            d_t_t1 += p_t + math.log2(p_t/p_t1) 
          
    d_t1_t = 0
    for hash,eq in eqcsP.items():
        p_t = len(eq.members)/num_verticesP
        if hash in eqcsC.keys():
            p_t1 = len(eqcsC[hash].members)/num_verticesC
            d_t1_t += p_t1 + math.log2(p_t1/p_t)
    return d_t_t1+d_t1_t
    


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str,dest='config', help='path to config file')  
    args = parser.parse_args()

    cfg = configparser.ConfigParser()
    cfg.read(args.config)
    
    
    base_dirC = cfg_u.makePath(cfg['DataExchange']['basedirC'])
    base_dirP = cfg_u.makePath(cfg['DataExchange']['basedirP'])
    
    
    gsC = None
    gsP = None
    
    file_nameC = str(cfg_u.makePath(cfg['GraphSummary']['save_fileC']))
    file_nameP = str(cfg_u.makePath(cfg['GraphSummary']['save_fileP']))
    print("Graphsummary file:", file_nameC)

    graph_dataC =  base_dirC/ file_nameC
    graph_dataP =  base_dirP/ file_nameP
    
    gsC = gsg.graph_for_summary()
    gsC.load( graph_dataC )
    
    gsP = gsg.graph_for_summary()
    gsP.load( graph_dataP )
    
    print("Graphs are loaded")
    
    #file
    if not  os.path.exists(base_dirC / "binary_measures"):
        os.makedirs(base_dirC / "binary_measures")
    name = base_dirC / "binary_measures"/"binary_measures.txt"
    f = open(name,"w+")
    
    #1AC
    print("\n1-hop Attribute Collection:")
    f.write("1-hop Attribute Collection:\n")
    #Attribute Collection
    v_t = gsP.current_eqcs1
    v_t1 = gsC.current_eqcs1
    
    jacV = 1 - (len(v_t1.intersection(v_t)) /len(v_t1 | v_t))
    print("JacV = "+str(jacV))
    f.write("JacV = "+str(jacV)+"\n")

    num_verticesC1 = gsC.num_vertices
    eqcsC1 = gsC.eqcs1
    num_verticesP1 = gsP.num_vertices
    eqcsP1 = gsP.eqcs1
    kld = KLD(num_verticesC1,eqcsC1, num_verticesP1,eqcsP1)
    print("Kullback–Leibler Divergence = "+str(kld))
    f.write("Kullback–Leibler Divergence = "+str(kld)+"\n")

    
    #2AC
    print("\n2-hop Attribute Collection")
    f.write("\n2-hop Attribute Collection\n")

    v_t = gsP.current_eqcs2
    v_t1 = gsC.current_eqcs2
    
    jacV = 1 - (len(v_t1.intersection(v_t)) /len(v_t1 | v_t))
    print("JacV = "+str(jacV))
    f.write("JacV = "+str(jacV)+"\n")

    num_verticesC2 = gsC.num_vertices
    eqcsC2 = gsC.eqcs2
    num_verticesP2 = gsP.num_vertices
    eqcsP2 = gsP.eqcs2
    kld = KLD(num_verticesC2,eqcsC2, num_verticesP2,eqcsP2)
    print("Kullback–Leibler Divergence = "+str(kld))
    f.write("Kullback–Leibler Divergence = "+str(kld)+"\n")

    
   
    
    
    
if __name__ == "__main__":
    main()    
