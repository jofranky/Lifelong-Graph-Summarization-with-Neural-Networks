"""
config_util.py: Wrapper functions for the modules configparser and pathlib.
"""

import configparser
import pathlib



def readConfig(sectionKey, readAsPath=True, section = 'DyldoFiles'):
    """
    This function returns the value of the given key as String

    
    Args:
        sectionKey (str): The key of the section
        readAsPath (bool): If the corresponding value should be read as path ( default is True )
        section (str): The section ( default is 'DyldoFiles' )
    """
    config = configparser.ConfigParser()
    #checks if section and key are existing
    if config.has_option(section, sectionKey):
        if not readAsPath:
            return config[section][sectionKey]
        else:
            return makePath(config[section][sectionKey])
    else:
        return 'error: section \'' + section + '\' or key \'' + sectionKey + '\' does not exist'
    


def makePath(fileDirectoryAndName, fileNameSuffix = ''):
    """
    This function creates a PurePath and has the option to add a file name suffix to files with six digit fileendings (i.e. '.nq.gz')

    
    Args:
        fileDirectoryAndName (str): File directory and name
        fileNameSuffix (str): A file name suffix to files with six digit fileendings ( default is '' )
    """
    if fileNameSuffix:
        return pathlib.PurePath(str(fileDirectoryAndName)[0:-6] + fileNameSuffix + str(fileDirectoryAndName)[-6:])
    else:
        return pathlib.PurePath(fileDirectoryAndName)