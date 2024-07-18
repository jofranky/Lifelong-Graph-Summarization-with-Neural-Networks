import pickle
import os.path as osp

import site
import sys

site.addsitedir('../../lib')  # Always appends to end

#print(sys.path)

import timeit

import pathlib
from tqdm import tqdm


import rdflib
import rdflib.parser as rdflib_parser
import rdflib.plugins.parsers.ntriples as triples
import rdflib.plugins.parsers.nquads as quads



class vertex_graph():
#( 1 = attribute collection; 2 = class  collection) 
   """
   A class to represent a vertex of the RDF graph that contains information for the sampling and summary
   """
   def __init__(self, hash1, changes1,hash2, changes2 ):
       """
       Initialize the subject_information object for all summaries
       """
       self.edges = [] 
       self.degree = 0
       self.properties = []
       self.hop1 = set()
       self.hop2 = set()
       
       #for  1-hop attribute collection 
       self.last_hash1 = hash1
       self.hash1 = float("NaN") #EQCs
       self.eqcs1Index = 0 #Index of the EQCs
       self.changes1 = changes1 #how often did the vertex change the EQCs #meta
       
       #for  2-hop attribute collection 
       self.last_hash2 = hash2
       self.hash2 = float("NaN") #EQCs
       self.eqcs2Index = 0 #Index of the EQCs
       self.changes2 = changes2 #how often did the vertex change the EQCs #meta
      
       
 
class predicate_graph():
    def __init__(self,first):
       self.first_discovery = first #meta
       self.last_discovery = first #meta
       self.frequency = 0 # in current_features it is the frequency in current graph and in all_features it is the frequency in prev. graph and current graph
class eqcs():
     """
     first all vertices then the eqcs are created for statisitics
     ps: list of properties
     id of eqcs according to unique_subjects 
     type: indicates the summary type
     """
     def __init__(self, s, ps,type,id):
        self.type = type
        self.members = set()
        self.members.add(s)
        self.id = id
        self.degree = len(ps) 
        self.edgesP = ps

        self.weight = 0

class t_graph():
    def __init__(self,current_subjects,feature_index,feature_list,label_list1,label_list2):
        self.current_subjects  = current_subjects       # Key-Value Store of current subjects with the subject/vertex as key and the subject object as value (includes hash and index of  eqcs) . For spliting and calculating the weighs. 
        #1AC
        self.size1=  len(label_list1) #get size of Output vector
        #2AC
        self.size2=  len(label_list2) #get size of Output vector
        
        self.feature_index = feature_index #Key-Value for getting the index of a predicate
        self.sizeFeature=  len(feature_list )#Get size of Input vector
        
        
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
  
class graph_for_summary():
    """
    A class used to calculate graph summaries
    """
   
    def __init__(self,prev = None):
        print(prev == None)
        if prev == None:
            
            #for getting the index of an eqcs
            self.label_ids1= {}   
            self.label_ids2= {}     
            
            #graph information
            self.num_vertices = 0 #current number vertices
            self.num_features = 0#self.num_features = 0 #current number of predeciate
            self.count_edge = 0
            self.index = 0                  # index of newest subject

            self.time = 1 #time for lifelong learning starting by 1
            
            
            #subjects and objects ids, only vertices with at least one edge are considered for the summaries
            self.current_subjects = {}#self.graph_information = {}     # Key-Value Store of current subjects with the subject/vertex as key and the subject as value
            self.current_subjects_set = set() #for testing if seen at this snapshot
            
            self.unique_subjects = []       # List is used to give a subject its index
            self.all_subjects  = {}       # Key-Value Store of all subjects with the index of the subject/vertex as key and the subject as value
            self.all_subjects_set  = set() #for testing if seen at all
            
            #predeciates and types
            self.current_features = {} # Key-Value Store of current predeciates with the index of the predeciate as key and the predicate  as value
            self.current_features_set = set()
            self.all_features = {} # Key-Value Store of all predeciates  with the index of the predeciate as key and the predicate  as value
            self.all_features_set = set()
            self.feature_list = []          # unique predicates for all summaries , to give index to predeciates saved for all graphs
            self.feature_index = {}  #Key-Value for getting the index of a predicate
            self.f_index = 0
            
            #EQCs  for  1-hop attribute collection 
            self.label_dict1 = {}   #for training  # key: hash value: list( subjects )
            self.label_list1 = []   #id for all eqcs
            self.eqcs1= {}      # Key-Value Store  with the hash of eqcs as key and the eqcs  as value
            self.current_eqcs1Edges = set()  #for binary measures
            self.current_eqcs1 = set() #current EQCs
            self.first_eqcs1 = set()  #EQCs of the first snapshot - Hashes
            self.prev_eqcs1 = set() #EQCs of the previous snapshot - Hashes
            
            #EQCsf or  2-hop attribute collection 
            self.label_dict2 = {}   #for training  # key: hash value: list( subjects )
            self.label_list2 = []   #id for all eqcs
            self.eqcs2= {}      # Key-Value Store with the hash of eqcs as key and the eqcs  as value
            self.current_eqcs2Edges = set()  #for binary measures
            self.current_eqcs2 = set() #current EQCs
            self.first_eqcs2 = set()  #EQCs of the first snapshot - Hashes
            self.prev_eqcs2 = set() #EQCs of the previous snapshot - Hashes 
            
        else:
            print("else")
            #for getting the index of an eqcs
            self.label_ids1= {}   
            self.label_ids2= {}     
            #graph information
            self.num_vertices = 0 #current number vertices
            self.num_features = 0#self.num_features = 0 #current number of predeciate
            self.count_edge = 0

            self.time = prev.time +1
            #subjects
            self.current_subjects = {}#self.graph_information = {}     # Key-Value Store of current subjects with the index of the subject/vertex as key and the subject as value
            self.current_subjects_set = set() #for testing if seen at this snapshot
            
            self.unique_subjects = prev.unique_subjects       # List is used to give a subject its index
            self.all_subjects  = prev.all_subjects       # Key-Value Store of all subjects with the index of the subject/vertex as key and the subject as value
            self.all_subjects_set  = prev.all_subjects_set #for testing if seen at all
            
            #predeciates
            self.current_features = {} # Key-Value Store of current predeciates with the index of the predeciate as key and the predicate  as value
            self.current_features_set = set()
            self.all_features = prev.all_features# Key-Value Store of all predeciates  with the index of the predeciate as key and the predicate  as value # Here frequency means frequency in all previous and current graph
            self.all_features_set = prev.all_features_set
            self.feature_list =  prev.feature_list         # unique predicates for all summaries , to give index to predeciates saved for all graphs
            self.feature_index = prev.feature_index   #Key-Value for getting the index of a predicate
            self.f_index = prev.f_index
            
            
            
            #EQCs for  1-hop attribute collection 
            self.label_dict1 = {}   #for training  # key: hash value: list( subjects )
            self.label_list1 = prev.label_list1   #id for all eqcs
            self.eqcs1= {}      # Key-Value Store  with the hash of eqcs as key and the eqcs  as value
            
            self.current_eqcs1Edges = set()  #for binary measures
            self.current_eqcs1 = set() #current EQCs#meta
            if prev.time == 1:
               self.first_eqcs1  = prev.current_eqcs1 #meta
               self.prev_eqcs1  = prev.current_eqcs1 #meta
            else:  
                self.first_eqcs1  = prev.first_eqcs1 #meta
                self.prev_eqcs1  = prev.current_eqcs1 #meta

            
            #EQCs for  2-hop attribute collection 
            self.label_dict2 = {}   #for training  # key: hash value: list( subjects )
            self.label_list2 = prev.label_list2   #id for all eqcs
            self.eqcs2= {}      # Key-Value Store with the hash of eqcs as key and the eqcs  as value
            
            self.current_eqcs2Edges = set()  #for binary measures
            self.current_eqcs2 = set() #current EQCs#meta 
            if prev.time == 1:
               self.first_eqcs2  = prev.current_eqcs2#meta
               self.prev_eqcs2  = prev.current_eqcs2#meta
            else:  
                self.first_eqcs2  = prev.first_eqcs2#meta
                self.prev_eqcs2  = prev.current_eqcs2#meta

            
    
    def is_rdf_type( self, s ):
        """
        Check if the given string contains rdf and therefore is an rdf-type

        
        Args:
            s (str): Feature used by the graph summary model

        Returns:
            bool: If it is a rdf-type
        """
        return "http://www.w3.org/1999/02/22-rdf-syntax-ns#type" in s 
    
    
    def create_graph_information( self, path):
        """
        Calculates the needed data to train/test the networks and analyzing the summaries

        
        Args:
            path (str): path to the nq-file
        """
        
        with open(path) as f:
            number_lines = sum(1 for _ in f)
        
        print("Files has",number_lines,"lines.")
        
        
        
        with open(path) as f:
            count_line = 0
            count_invalid = 0

            parseq = quads.NQuadsParser() 
            sink = rdflib.ConjunctiveGraph()
            while True:
                # Get next line from file
                line = f.readline()
             
                # if line is empty
                # end of file is reached
                if not line:
                    break
                
                count_line += 1
                
                if count_line % 10000 == 0:
                    print("Read line", count_line, "of", number_lines, "(", count_line / number_lines * 100.0,"%)")
                        
                sink = rdflib.ConjunctiveGraph()   
                strSource = rdflib_parser.StringInputSource(line.encode('utf-8'))

                try:
                    #try parsing the line to a valid N-Quad
                    parseq.parse(strSource, sink)
                    
                    self.count_edge += 1
                    
                    #write the validated N-Quad into the filtered File
                    
                    #print( list(sink.subjects()),list(sink.predicates()),list(sink.objects() ) )
                    s = str(list(sink.subjects())[0])
                    p = str(list(sink.predicates())[0])
                    o = str(list(sink.objects())[0]) # is also a subject. In worst case, it has no edges
                    
                    if not self.is_rdf_type(p):  #ignor type              
                        #for current subject                                      
                        if( s in self.current_subjects_set ):
                            self.current_subjects[s].edges.append((p, o))
                            self.current_subjects[s].degree += 1
                            self.all_subjects[s].edges.append((p, o))
                            self.all_subjects[s].degree += 1
                        elif (s in self.all_subjects_set):    
                            self.current_subjects_set.add(s)
                            hash1 = self.all_subjects[s].hash1
                            hash2 = self.all_subjects[s].hash2
                            changes1 = self.all_subjects[s].changes1
                            changes2 = self.all_subjects[s].changes2
                            self.current_subjects[s] = vertex_graph( hash1, changes1,hash2, changes2 )
                            self.all_subjects[s] = vertex_graph( hash1, changes1,hash2, changes2 )
                            self.current_subjects[s].edges.append((p, o))
                            self.current_subjects[s].degree += 1
                            self.all_subjects[s].edges.append((p, o))
                            self.all_subjects[s].degree += 1
                            self.num_vertices += 1 
                        else:   
                            self.unique_subjects.append(s)
                            self.all_subjects_set.add(s)
                            self.all_subjects[s] = vertex_graph(float("NaN"),0,float("NaN"),0,float("NaN"), 0 )
                            self.all_subjects[s].edges.append((p, o))
                            self.all_subjects[s].degree += 1
                            self.current_subjects_set.add(s)                    
                            self.current_subjects[s] = vertex_graph(float("NaN"),0,float("NaN"),0,float("NaN"), 0 )
                            self.current_subjects[s].edges.append((p, o))
                            self.current_subjects[s].degree += 1
                            self.num_vertices += 1

                        #add to list
                        if (p not in self.current_subjects[s].properties):#no duplicates
                            self.current_subjects[s].properties.append(p)
                            self.all_subjects[s].properties.append(p)


                        if( p not in self.all_features_set): #only add a predeciate if it is important for the summary
                            self.feature_list.append( p )
                            self.feature_index[p] = self.f_index  #Key-Value for getting the index of a predicate
                            self.f_index += 1
                            self.all_features[p] = predicate_graph(self.time)
                            self.current_features[p]= predicate_graph(self.time)
                            self.all_features_set.add(p)
                            self.current_features_set.add(p)
                            self.num_features += 1


                        if(p not in  self.current_features_set): #only add a predeciate if it is important for the summary
                           first =  self.all_features[p].first_discovery
                           self.current_features[p] = predicate_graph(self.time) 
                           self.current_features_set.add(p)
                           self.current_features[p].first_discovery = first
                           self.num_features += 1


                        self.all_features[p].frequency += 1
                        self.current_features[p].frequency  += 1

                        self.all_features[p].last_discovery  =  self.time
                        self.current_features[p].last_discovery   = self.time
                    
                  
                except triples.ParseError:
                    #catch ParseErrors and write the invalidated N-Quad into the trashed File
                    count_invalid += 1
                    
                    #print the number of Errors and current trashed line to console
                    print('Wrong Line Number ' + str(f'{count_invalid:,}') + ': ' + line)
                
            print("lines read:", count_line)
            print("invalid lines read:", count_invalid)
            
    
    
    
          
            
    def calculate_graph_summary( self ):
        """
        Calculates all summaries
        """
        
        #1-hop
        h1s = {}
        print("1-hop")
        for gi in  tqdm(self.current_subjects.items()):
            edges = gi[1].edges
            for e in edges:
                gi[1].hop1.add((e[0],0))
            h1 = 0    
            for (p,h) in gi[1].hop1:
                h1 = h1 ^ hash( str(p)+str(h) )
            h1s[gi[0]] = h1
            tmp_hash1 = h1
            
            former_hash1 = gi[1].last_hash1
            gi[1].hash1 = tmp_hash1
            
            if(tmp_hash1 != former_hash1):
                gi[1].changes1 += 1
            
            s = gi[0]
            self.all_subjects[gi[0]].hash1 = tmp_hash1
            self.all_subjects[gi[0]].changes1 = gi[1].changes1
        
            if tmp_hash1 not in self.label_list1:
                self.label_list1.append(tmp_hash1)

            if tmp_hash1 not in self.current_eqcs1:
                self.current_eqcs1.add(tmp_hash1)

            if tmp_hash1 in self.label_dict1:
                self.label_dict1[tmp_hash1].append(s)
                self.eqcs1[tmp_hash1].members.add(s)
            else:
                self.label_dict1[tmp_hash1] = [s]
                indexE = self.label_list1.index(tmp_hash1)
                self.eqcs1[tmp_hash1] = eqcs(s,gi[1].hop1,1,indexE) 
                self.label_ids1[tmp_hash1]  =  indexE 
                
                
            gi[1].eqcs1Index = self.label_ids1[tmp_hash1]  
            
        #2-hop
        h2s = {}
        print("2-hop")
        for gi in  tqdm(self.current_subjects.items()):
            edges = gi[1].edges
            for e in edges:
                if e[1] in h1s.keys():
                    gi[1].hop2.add((e[0],h1s[e[1]] ))
                else:
                    gi[1].hop2.add((e[0],0 ))
            h2 = 0    
            for (p,h) in gi[1].hop2:
                h2 = h2 ^ hash( str(p)+str(h) )
            h2s[gi[0]] = h2
            tmp_hash2 = h2
            
            former_hash2 = gi[1].last_hash2
            gi[1].hash2 = tmp_hash2
            
            if(tmp_hash2 != former_hash2):
                gi[1].changes2 += 1
            
            s = gi[0]  
            self.all_subjects[gi[0]].hash2 = tmp_hash2
            self.all_subjects[gi[0]].changes2 = gi[1].changes2
        
            if tmp_hash2 not in self.label_list2:
                self.label_list2.append(tmp_hash2)

            if tmp_hash2 not in self.current_eqcs2:
                self.current_eqcs2.add(tmp_hash2)

            if tmp_hash2 in self.label_dict2:
                self.label_dict2[tmp_hash2].append(s)
                self.eqcs2[tmp_hash2].members.add(s)
            else:
                self.label_dict2[tmp_hash2] = [s]
                indexE = self.label_list2.index(tmp_hash2)
                self.eqcs2[tmp_hash2] = eqcs(s,gi[1].hop2,2,indexE)
                self.label_ids2[tmp_hash2]  =  indexE 

                    
            gi[1].eqcs2Index = self.label_ids2[tmp_hash2]  
            
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


     # get informations
    def get_num_features( self ):
        """
        Return number of features

        Returns:
            int: Number of features
        """
        return len( self.feature_list )
        
    def get_num_classes1( self ):
        """
        Return number of classes

        Returns:
            int: Number of classes
        """
        return ( len( self.current_eqcs1 ) )
        
    def get_num_classes2( self ):
        """
        Return number of classes

        Returns:
            int: Number of classes
        """
        return ( len( self.current_eqcs2 ) )
        
    
    def get_num_vertices( self ):
        """
        Return number of vertices

        Returns:
            int: Number of vertices
        """
        return self.num_vertices
       
    def createT(self,filename):
        """
        For creating  a file for the creation of the subgraphs for training/testing
        """
        tg = t_graph(self.current_subjects,self.feature_index,self.feature_list,self.label_list1,self.label_list2)
        tg.save(filename)
