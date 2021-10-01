"""
@Author: Gal Almog
Calculate split half rank correlations for each category
"""
import numpy as np
from numpy.core.numeric import NaN
import pandas as pd
from collections import Counter
from pathlib import Path
import random
from scipy.stats import spearmanr
import statistics
import sys

category = sys.argv[1]

#import experiment data
df = pd.read_csv("./memoir_experiment_data.csv", index_col=False)

# get memcat data for all our target images
targets = []
for path in Path("./memoir_targets").glob("**/*.jpg"):
    targets.append(path.name) #get list of all target image file names

df_memcat = pd.read_csv("./memcat_image_data.csv", index_col=False)
df_memcat = df_memcat.drop(df_memcat.columns[0], axis =1)
df_memcat = df_memcat.query("image_file in @targets") # query memcat data to get the data for our target images

# QUALITY ASSURANCE

# set the index to be the participant ids
df.set_index(keys=['participant_id'], drop=False,inplace=True)

# get a list of unique participants
participants=df['participant_id'].unique().tolist()

# remove participants with no recorded data
for participant in participants:
    p_df = df.loc[(df.participant_id==participant), 'participant_id'] #subset of df for each participant
    if len(p_df) <= 1:
        df = df.drop(participant, axis=0)

participants = df['participant_id'].unique().tolist()

indexes_to_drop = {}

for participant in participants:
    p_df = df.loc[df.participant_id==participant] #subset of df for each participant
    indexes_to_drop[participant] = []
    
    block = max((p_df['block_index']).astype(int)) # num blocks completed
    kicked = (p_df['kicked_out'])
    if (kicked.all() == 'completed'):
        next #if completed move on to next participant and don't remove any levels
    elif not kicked.isnull().values.all(): #if were kicked out, remove last level
        indexes_to_drop[participant].append(block)
    elif block < 10: # if completed between 1 and 10 blocks but quit voluntarily remove last level
        indexes_to_drop[participant].append(block)

# drop indexes from the dataframe
for ind in indexes_to_drop:
    if indexes_to_drop[ind]: # iterate through non-empty lists
        df = df[~((df.index == ind) & (df['block_index'].astype(int).isin(indexes_to_drop[ind])))]

participants = df['participant_id'].unique().tolist() #reset participants list
df_targets = df[(df['trial_type'] == 'target') | (df['trial_type'] == 'target repeat')]

# split each unique target image into df
dict_of_targets = {k: v for k, v in df_targets.groupby('image_file')}
num_targets = len(dict_of_targets)
list_of_targets = list(dict_of_targets.keys())

# function to split data in half randomly
def split(ls, n, seed):
    random.Random(seed).shuffle(ls)
    return [ls[i::n] for i in range(n)]

# function to calculate memorability scores and rank correlations with downsampling
def calculate_mem(participants_half, seed):

    results_list=[]
    for image_file, values in dict_of_targets.items(): #values is a df

        #Remove first viewing of targets if there was no second viewing
        values = values[values.index.isin(participants_half)] # split participant pool in half
        values = values[values.index.duplicated(False)] # only keeps the duplicated indexes (participants)
        values.index.name = None
        l = len(values['participant_id'].unique().tolist())
        
        for downsample in np.linspace(1,0.1, num=30): #downsample range
            values = pd.concat(group[1] for group in random.sample(list(values.groupby(['participant_id'], sort=False)), k = int(l*downsample)))
            
            measurements = Counter(values['response_type']) # calculate measurements for the target
            hm = (measurements['hit'] + measurements['miss'])
            fc = measurements['false alarm'] + measurements['correct rejection']
            h = measurements['hit']
            f = measurements['false alarm']

            mem_score_yfa =  ((h - f)/ hm) if hm!= 0 else 0
            mem_score_nfa = (h/hm) if hm!= 0 else 0
            for index, row in df_memcat.iterrows():
                if (image_file == row[0]) & (row[1] == category):
                    new = row.append(pd.Series([h, f, hm, mem_score_yfa, mem_score_nfa, downsample], index=['memoir_h','memoir_f','memoir_nresp','memoir_mem_score_yfa','memoir_mem_score_nfa', 'downsample']))
                    results_list.append(new)
        
    results = pd.DataFrame(results_list)
    results = results[['image_file','memoir_mem_score_nfa', 'memoir_nresp', 'downsample']]
    return results

rank_corr_results = []

for i in range(25):
    # create split half dfs
    first, second = split(participants, 2, i)

    # calculate memorability in 2 groups
    results_first = calculate_mem(first, i)
    results_second = calculate_mem(second, i)

    # group by each downsampling level
    for downsample in np.linspace(1,0.1, num=30):
        rf = results_first[results_first['downsample'] == downsample]
        rs = results_second[results_second['downsample'] == downsample]

        results = pd.merge(rf, rs, how='inner', on='image_file', suffixes=['_first','_second'])
        corr = spearmanr(results['memoir_mem_score_nfa_first'], results['memoir_mem_score_nfa_second'])[0]
        
        # mean nresp across all target images for this downsampling level (1 number)
        nresp = statistics.mean(results['memoir_nresp_first'] + results['memoir_nresp_second'])
        
        # half_split_res.append([corr, nresp, downsample])
        rank_corr_results.append([corr, nresp, downsample])
        print(corr, nresp, downsample, i)

rank_corr_df = pd.DataFrame(rank_corr_results, columns=['corr', 'nresp', 'downsample'])
rank_corr_df.to_csv('./rank_corr_results_%s_nfa.csv' % category, index=False)

final_results = rank_corr_df.groupby('downsample').mean()
final_results.to_csv('./rank_corr_results_means_%s_nfa.csv' % category, index=False)