import numpy as np
import csv
import os
import sys


def get_params(model_id):
    """
    read all parameters for specified model id

    model_id : int
        specifies which column in the csv file is used
    """
    
    integerParams = ['general_populationSize', 'GPeArkyCopy_On', 'threads']
    string_params = ['general_id']

    csvPath = os.path.dirname(os.path.realpath(__file__))+'/parameters.csv'
    csvfile = open(csvPath, newline='')

    params = {}
    reader = csv.reader(csvfile, delimiter=',')
    fileRows = []
    idx = -1
    ### check if model_id is in the .csv file
    for row in reader:
        fileRows.append(row)
        if 'general_id'==row[0] and True in [model_id == row[i] for i in range(1,len(row))]:
            idx = [model_id == row[i] for i in range(1,len(row))].index(True)+1
        elif 'general_id'==row[0]:
            print('No Parameters available for given parameter ID '+model_id+'! (file '+csvPath+')')
            quit()
    if idx==-1:
        print('No general_id in parameter csv file!')
        quit()
    ### read the column corresponding to model_id
    for row in fileRows:
        if '###' in row[0]: continue
        if row[0] in integerParams:
            params[row[0]] = int(float(row[idx]))
        elif row[0] in string_params:
            params[row[0]] = row[idx]
        else:
            params[row[0]] = float(row[idx])

    csvfile.close()
    
    ### ADD additional params
    params['toRGB']                 = {'blue':np.array([3,67,223])/255., 'cyan':np.array([0,255,255])/255., 'gold':np.array([219,180,12])/255., 'orange':np.array([249,115,6])/255., 'red':np.array([229,0,0])/255., 'purple':np.array([126,30,156])/255., 'grey':np.array([146,149,145])/255., 'light brown':np.array([173,129,80])/255., 'lime':np.array([170,255,50])/255., 'green':np.array([21,176,26])/255., 'yellow':np.array([255,255,20])/255., 'lightgrey':np.array([216,220,214])/255.}
    params['Fig7_order']            = ['GPeArky', 'StrD1', 'StrD2', 'STN', 'cortexGo', 'GPeCp', 'GPeProto', 'SNr', 'Thal', 'cortexStop', 'StrFSI']
    params['titles_Code_to_Script'] = {'cortexGo':'cortex-Go', 'cortexStop':'cortex-Stop', 'cortexPause':'cortex-Pause', 'StrD1':'StrD1', 'StrD2':'StrD2', 'StrFSI':'StrFSI', 'GPeProto':'GPe-Proto', 'GPeArky':'GPe-Arky', 'GPeCp':'GPe-Cp', 'STN':'STN', 'SNr':'SNr', 'Thal':'thalamus', 'IntegratorGo':'Integrator-Go', 'IntegratorStop':'Integrator-Stop'}

    
    return params




