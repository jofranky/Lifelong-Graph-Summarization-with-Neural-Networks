import argparse
import configparser

import re
import gzip

import rdflib
import rdflib.parser as rdfParser
import rdflib.plugins.parsers.ntriples as triples
import rdflib.plugins.parsers.nquads as quads

import site

site.addsitedir('../../lib')  # Always appends to end
from config_utils import config_util as cfg_u


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str,dest='config', help='path to config file')  #to parse the confi file (config_filterRDF.ini)
    args = parser.parse_args() #get different sections of config
    
    cfg = configparser.ConfigParser()
    cfg.read(args.config) #parse the arguments of the different section

    if cfg.getboolean('Dyldo', 'finished'):
        print("File is probably already filtered. If not set finished=False and check the other arguments")
        return

    
    """
  
    Args:
        fileDirectoryAndName (pathlib.PurePath): Path to nquads file
        writeFileName (pathlib.PurePath): Path for the validated and filtered data
        trashFileName (pathlib.PurePath): Path for the filtered trash
        preSkolemize (bool): preSkolemize is a Boolean which replaces all blanknodes with handmade IRIs on True ( default is False )
        numLines (int): numLines is the number of lines which shall be checked in the input file, numLines == 0 sets the number to the the number of lines in the file ( default in 0 )
    """
    
    base_dir = cfg_u.makePath(cfg['DataExchange']['basedir'])#data location

    
    fileDirectoryAndName  =  base_dir.joinpath( cfg_u.makePath(cfg['Dyldo']['raw_datafile']) ) #raw datafile 
    writeFileName =  base_dir / cfg_u.makePath(cfg['Dyldo']['filtered_datafile'])#filtered datafile
    trashFileName =  base_dir / cfg_u.makePath(cfg['Dyldo']['trashed_datafile'])#trashed datafile
    numLines =  cfg.getint('Dyldo', 'num_lines')
    preSkolemize =  cfg.getboolean('Dyldo', 'pre_skolemize')
    
    #it counts the line if not already counted
    if cfg.getboolean('Dyldo', 'nun_lines_counted') == False:
        numLines = sum(1 for line in gzip.open(cfg_u.makePath(fileDirectoryAndName), 'rt', encoding = 'utf-8'))
        cfg.set('Dyldo', 'num_lines',str(numLines))
        cfg.set('Dyldo', 'nun_lines_counted', "True")
        configfile = open(args.config,'w') #to write config back
        cfg.write(configfile)
        configfile.close()
        
    #filtering begins
    begin_line = cfg.getint('Dyldo',"begin_line") #Beginning at begin_line
    read_lines = cfg.getint('Dyldo',"read_lines") #How many lines should be read if possible
    
    #Determine end for this iteration of filtering
    end_line = 0
    if (begin_line + read_lines) > numLines:
        end_line = numLines+1
    else:
        end_line = begin_line + read_lines 
    
    #read data
    readFile = gzip.open(fileDirectoryAndName, 'rt', encoding = 'utf-8') #gzip needs specified text mode ('t') if encoding gets used
    
    mode = 'wt+'
    #Determine if files have to be created (wt) or not (at)
    if(begin_line == 1):
        mode = 'wt+'
        print("Starting with filtering and create files")
    else:
        mode = 'at+'
        print("Continue with filtering at line "+ str(begin_line))
        print("Have to re-read file before continuing with filtering")
        #re-read file to continue
        for i in range(begin_line-1): 
            if( i%100000 == 0):
                print(str(i) + " of " + str(begin_line-1) + " lines.")  
            readFile.readline()
        print("Finished with re-reading")
    
    writeFile = open(writeFileName, mode,encoding="utf-8" )
    trashFile = open(trashFileName, mode,encoding="utf-8" )
        
    #generate the sink which is used to test the quads
    sink = rdflib.ConjunctiveGraph()
    errNo = 0
    ignoreNo = 0
    
    for i in range(begin_line, end_line):        
        #progress output
        if( i%100000 == 0 ):
           print(str(f'{i:,}') + " of " + str(f'{end_line-1:,}') + " lines.")            
        if( i%1000000 == 0 ):
           del sink
           sink = rdflib.ConjunctiveGraph()
           #cache last line (it could crash because of memory)
           cfg.set('Dyldo', 'begin_line',str(i))
           configfile = open(args.config,'w') #to write config back
           cfg.write(configfile)
           configfile.close()
           
        
        
        line = readFile.readline()
        
        if preSkolemize:
            
            #splitting at non escaped " (preparation for preskolemization) 
            quotationSplitLine = re.split(r'(?<!\\)\"', line) #(negative lookbehind regex in order to respect escaped " in literals)
            
            #handle blanknodes
            if '_:' in line:
                if(len(quotationSplitLine) == 1): #no literal as object in line
                    line = manageBNode_(line)
                elif (len(quotationSplitLine) == 3): #literal as object in line
                    line = ''
                    for j in range(0, len(quotationSplitLine)):
                        if(j%2 == 1): #literals
                            line = line + '"' + quotationSplitLine[j] + '"'
                        else: #non-literals
                            line = line + manageBNode_(quotationSplitLine[j])
                else:
                    #failed to eat line with literal, therefore emptying it on order to ignore it completely
                    ignoreNo += 1
                    print('Ignored Line Number ' + str(f'{ignoreNo:,}') + ': ' + line)
                    line = ''
                    
        parseq = quads.NQuadsParser()
        strSource = rdfParser.StringInputSource(line.encode('utf-8'))
        
        try:
            #try parsing the line to a valid N-Quad
            parseq.parse(strSource, sink) #Here MemoryError
            
            #write the validated N-Quad into the filtered File
            writeFile.write(line) #maybe flush

        except triples.ParseError:
            #catch ParseErrors and write the invalidated N-Quad into the trashed File
            trashFile.write(line)
            
            #print the number of Errors and current trashed line to console
            errNo += 1
            print('Wrong Line Number ' + str(f'{errNo:,}') + ': ' + line)
    
    
    #close all Filereaders
    readFile.close()
    writeFile.close()
    trashFile.close()
    
    if (ignoreNo > 0):
        print('Total Number of wrong N-Quads: ' + str(f'{errNo:,}') + ' and total Number of ignored N-Quads: ' + str(f'{ignoreNo:,}'))
    else:
        print('Total Number of wrong N-Quads: ' + str(f'{errNo:,}'))
    
    #for next iteration
    cfg.set('Dyldo', 'begin_line',str(end_line))
    configfile = open(args.config,'w') #to write config back
    cfg.write(configfile)
    configfile.close()
    if(end_line == numLines+1):
        print("Finished with filtering file. Please eliminate all duplicates.")
        cfg.set('Dyldo', 'finished','True')
        configfile = open(args.config,'w') #to write config back
        cfg.write(configfile)
        configfile.close()
    else:
        print(str(f'{end_line-1:,}') + " of " + str(f'{numLines:,}') + " lines are already filtered.\nPlease rerun programm, so that the whole data can be filtered") 

        


def manageBNode_(line):    
    """
    This function replaces all blanknodes in the line with a handmade IRI (prepend 'https://blanknode/', delete the '_:' and append '>')

    
    Args:
        line (str): Line to manage
    """
    line = re.sub('( |^)_:', '\g<1><https://blanknode/', line)
    line = re.sub("(( |^)[^@ ]*[^^,>\" ]) ", "\g<1>> ", line)
    
    return line




if __name__ == "__main__":
    main()
    
