import numpy as np
import csv
import os
import sys
dir_path = os.path.dirname(os.path.realpath(__file__))

curr_ID = 8007

def readParams(csvPath,param_ID,integerParams):
    """
    read all parameters from specified csv file

    csvPath : str
        path to the .csv file 
    param_ID : int
        specifies which column in the csv file is used
    integerParams : list of str
        list of parameter names whose values should be integer instead of float
    """

    if sys.version_info[0]<3:
        csvfile = open(csvPath, 'rb')# Python 2
    else:
        csvfile = open(csvPath, newline='')# Python 3

    params = {}
    reader = csv.reader(csvfile, delimiter=',')
    fileRows = []
    idx = -1
    ### check if param_ID is in the .csv file
    for row in reader:
        fileRows.append(row)
        if 'general_id'==row[0] and True in [param_ID == int(float(row[i])) for i in range(1,len(row))]:
            idx = [param_ID == int(float(row[i])) for i in range(1,len(row))].index(True)+1
        elif 'general_id'==row[0]:
            print('No Parameters available for given parameter ID! (file '+__file__+')')
            quit()
    if idx==-1:
        print('No general_id in parameter file!')
        quit()
    ### read the column corresponding to param_ID
    for row in fileRows:
        if '###' in row[0]: continue
        if row[0] in integerParams:
            params[row[0]] = int(float(row[idx]))
        else:
            params[row[0]] = float(row[idx])

    csvfile.close()
    return params


### DEFINE PARAMETERS WITH OTHER DATATYPE THAN FLOAT
integerParams = ['general_populationSize', 'GPeArkyCopy_On', 'threads', 'general_id']

### GET PARAMETERS FROM CSV FILE
csvfilePath = dir_path+'/new_params.csv'
params = readParams(csvfilePath,curr_ID,integerParams)

### ADD SPECIFIC PARAMETERS
params['toRGB']                 = {'blue':np.array([3,67,223])/255., 'cyan':np.array([0,255,255])/255., 'gold':np.array([219,180,12])/255., 'orange':np.array([249,115,6])/255., 'red':np.array([229,0,0])/255., 'purple':np.array([126,30,156])/255., 'grey':np.array([146,149,145])/255., 'light brown':np.array([173,129,80])/255., 'lime':np.array([170,255,50])/255., 'green':np.array([21,176,26])/255., 'yellow':np.array([255,255,20])/255., 'lightgrey':np.array([216,220,214])/255.}
params['Fig7_order']            = ['GPeArky', 'StrD1', 'StrD2', 'STN', 'cortexGo', 'GPeCp', 'GPeProto', 'SNr', 'Thal', 'cortexStop', 'StrFSI']
params['titles_Code_to_Script'] = {'cortexGo':'cortex-Go', 'cortexStop':'cortex-Stop', 'cortexPause':'cortex-Pause', 'StrD1':'StrD1', 'StrD2':'StrD2', 'StrFSI':'StrFSI', 'GPeProto':'GPe-Proto', 'GPeArky':'GPe-Arky', 'GPeCp':'GPe-Cp', 'STN':'STN', 'SNr':'SNr', 'Thal':'thalamus', 'IntegratorGo':'Integrator-Go', 'IntegratorStop':'Integrator-Stop'}

