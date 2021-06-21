import pandas as pd
import numpy as np
from scipy import stats

data_path = 'data/'

def load_experimental_data():
    # Load dataframes

    # MartinPlatero plankton data

    df_ts = {}

    path = 'MartinPlatero/'
    files = ['41467_2017_2571_MOESM5_ESM_MartinPlatero_Plankton_Eukarya.csv',
             '41467_2017_2571_MOESM4_ESM_MartinPlatero_Plankton_Bacteria.csv']
    keys = ['plankton_eukarya', 'plankton_bacteria'] 

    for i, f in enumerate(files):
        x = pd.read_csv(data_path+path+f, na_values='NAN', index_col=0)
        x = x.iloc[:, :-1] # delete last columns which contains details on the otu's

        # only keep 200 most abundant species
        sum_x = x.sum(axis='columns')

        x = x[sum_x >= np.sort(sum_x)[-200]]

        x = x.T # species are in rows instead of columns

        x.insert(0, 'time', [int(j[4:7]) for j in x.index]) # day

        x = x.groupby('time').agg('mean').reset_index()

        x.columns = ['time'] + ['species_%d' % j for j in range(1, len(x.columns))]

        df_ts[keys[i]] = x


    # David stool data

    files = [data_path + 'Faust/25_timeseries/25_timeseries.txt', data_path + 'Faust/28_timeseries/28_timeseries.txt']
    keys = ['David_stool_A', 'David_stool_B']

    for i, f in enumerate(files):
        x = pd.read_csv(f, na_values='NAN', delimiter='\t', header=None)

        x = x.T

        x.insert(0, 'time', range(len(x)))

        x.columns = ['time'] + ['species_%d' % j for j in range(1, len(x.columns))]

        df_ts[keys[i]] = x

    # Caporaso body sites data

    sites = ['F4_L_palm_L6', 'F4_tongue_L6']

    for site in sites:
        file = data_path + 'Caporaso/' + site + '.txt'
        key = 'Caporaso_' + site

        x = pd.read_csv(file, delimiter='\t', skiprows=1, index_col=0, header=None)
        
        x.index = ['time'] + ['species_%d' % j for j in range(1, len(x.index))]

        x = x.T

        # only keep 200 most abundant species
        if len(x.columns) > 201:
            sum_x = x.sum(axis='rows')

            sum_x['time'] = np.inf

            sum_x.sort_values(ascending=False, inplace=True)

            x = x[sum_x.index.tolist()[:201]]

        x.columns = ['time'] + ['species_%d' % j for j in range(1, len(x.columns))]

        df_ts[key] = x
    return df_ts

def load_experimental_abundances():
    path = 'Arumugam/'
    files = ['MetaHIT_41SangerSamples.genus.csv', #'MetaHIT_41SangerSamples.phylum.csv',
              'MetaHIT_85IlluminaSamples.genus.csv', 'Turnbaugh_154Pyroseq16S.genus.csv']
    titles = ['Sanger', #'Sanger phylum', 
              'Illumina', 'Pyroseq']

    df_st = {title : pd.read_csv(data_path + path + file, index_col=0).transpose() for title, file in zip(titles, files)}
    return df_st

