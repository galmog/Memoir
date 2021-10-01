"""
@Author: Gal Almog
Calculate Memorability Scores
"""
import numpy as np
from numpy.core.numeric import NaN
import pandas as pd
from collections import Counter
from pathlib import Path

#import experiment data
df = pd.read_csv("./memoir_experiment_data.csv", index_col=False)

# get memcat data for all our target images
targets = []
for path in Path("./memoir_targets").glob("**/*.jpg"):
    targets.append(path.name) #get list of all target image file names

df_memcat = pd.read_csv("./memcat_image_data.csv", index_col=False)
df_memcat = df_memcat.drop(df_memcat.columns[0], axis =1)
df_memcat = df_memcat.query("image_file in @targets") # query memcat data to get the data for our target images


# select all target images (remove fillers)
df_targets = df[(df['trial_type'] == 'target') | (df['trial_type'] == 'target repeat')]

# split each unique target image into df
dict_of_targets = {k: v for k, v in df_targets.groupby('image_file')}
num_targets = len(dict_of_targets)

resp_greater = 0
resp_negative = 0
results_list=[]
for image_file, values in dict_of_targets.items(): #values is a df

    #Remove first viewing of targets if there was no second viewing
    values = values[values.index.duplicated(False)] # only keeps the duplicated indexes (participants)

    measurements = Counter(values['response_type']) # calculate measurements for the target
    hm = (measurements['hit'] + measurements['miss'])
    fc = measurements['false alarm'] + measurements['correct rejection']
    h = measurements['hit']
    f = measurements['false alarm']
    # calculate memorability scores
    mem_score_yfa =  ((h - f)/ hm) if hm!= 0 else 0
    mem_score_nfa = (h/hm) if hm!= 0 else 0

    # calculate response times:
    r_values = values.loc[values.button_press != 'NR']
    response_time = (pd.to_numeric(r_values['button_press']) - pd.to_numeric(r_values['trial_start']))
    resp_greater += sum(n > 1400 for n in response_time) # checking how many response times were greater/less than 1400ms
    resp_negative += sum(n < 0 for n in response_time)

    response_time = (response_time[(response_time < 1400) & (response_time > 0)]).mean()
    for index, row in df_memcat.iterrows():
        if image_file == row[0]:
            new = row.append(pd.Series([h, f, hm, mem_score_yfa, mem_score_nfa, response_time], index=['memoir_h','memoir_f','memoir_nresp','memoir_mem_score_yfa','memoir_mem_score_nfa', 'mean_response_time']))
            results_list.append(new)

results = pd.DataFrame(results_list)
results.to_csv('./memoir_results.csv', index=False) #save results