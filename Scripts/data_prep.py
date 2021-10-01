"""
@author: Gal Almog
Data cleaning and preparation, as well as quality assurance for experiment data.
"""
import pandas as pd
from numpy.core.numeric import NaN

"""
experiment data
"""
df1 = pd.read_csv("./memoryGameCsv-v1.csv", 
index_col=False, 
names = ['participant_id', 'loi', 'consent', 'gender', 'age', 'block_index',
       'trial_index', 'image_file', 'trial_type', 'response_type',
       'screen_width', 'screen_height', 'date', 'vm_count', 'fa_count', 'trial_start', 'button_press', 'kicked_out']) #experiment results
df1 = df1.iloc[101:] #remove testing

df2 = pd.read_csv("./memoryGameCsv-v2.csv", 
index_col=False, 
names = ['participant_id', 'loi', 'consent', 'gender', 'age', 'block_index',
       'trial_index', 'image_file', 'trial_type', 'response_type',
       'screen_width', 'screen_height', 'date', 'vm_count', 'fa_count', 'trial_start', 'button_press', 'kicked_out']) #experiment results
df2 = df2.iloc[13:] # remove testing

df = pd.concat([df1,df2])

# QUALITY ASSURANCE

# set the index to be the participant ids
df.set_index(keys=['participant_id'], drop=False,inplace=True)

# get a list of unique participants
participants=df['participant_id'].unique().tolist()

# remove participants with no recorded data
for participant in participants:
    p_df = df.loc[df.participant_id==participant] #subset of df for each participant
    if len(p_df) <= 1:
        df = df.drop(participant, axis=0)

participants = df['participant_id'].unique().tolist()

# quality exclusions 
    
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

df.to_csv('./memoir_experiment_data.csv', index=False)
