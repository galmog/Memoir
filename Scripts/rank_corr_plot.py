"""
@Author: Gal Almog
Rank correlation plot 
"""
from matplotlib.colors import is_color_like
import numpy as np
import pandas as pd
from collections import Counter
from pathlib import Path
from scipy.stats import ttest_rel, f_oneway, spearmanr
from scipy.stats.mstats_basic import pearsonr
from seaborn import palettes
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import seaborn as sns 
import matplotlib.pyplot as plt
import statistics
import statsmodels.api as sm
from statsmodels.formula.api import ols

# import all the data
df_all = pd.read_csv("./rank_corr_results_means.csv",
index_col=False, 
header = 0
)
df_all['Category'] = 'All'

df_all_nfa = pd.read_csv("./rank_corr_results_means_nfa.csv",
index_col=False, 
header = 0
)
df_all_nfa['Category'] = 'All'

df_sport = pd.read_csv("./rank_corr_results_means_sports.csv",
index_col=False, 
header = 0
)
df_sport['Category'] = 'Sport'
df_sport_nfa = pd.read_csv("./rank_corr_results_means_sports_nfa.csv",
index_col=False, 
header = 0
)
df_sport_nfa['Category'] = 'Sport'

df_animal = pd.read_csv("./rank_corr_results_means_animal.csv",
index_col=False, 
header = 0
)
df_animal['Category'] = 'Animal'
df_animal_nfa = pd.read_csv("./rank_corr_results_means_animal_nfa.csv",
index_col=False, 
header = 0
)
df_animal_nfa['Category'] = 'Animal'

df_vehicle = pd.read_csv("./rank_corr_results_means_vehicle.csv",
index_col=False, 
header = 0
)
df_vehicle['Category'] = 'Vehicle'
df_vehicle_nfa = pd.read_csv("./rank_corr_results_means_vehicle_nfa.csv",
index_col=False, 
header = 0
)
df_vehicle_nfa['Category'] = 'Vehicle'

df_food = pd.read_csv("./rank_corr_results_means_food.csv",
index_col=False, 
header = 0
)
df_food['Category'] = 'Food'
df_food_nfa = pd.read_csv("./rank_corr_results_means_food_nfa.csv",
index_col=False, 
header = 0
)
df_food_nfa['Category'] = 'Food'

df_landscape = pd.read_csv("./rank_corr_results_means_landscape.csv",
index_col=False, 
header = 0
)
df_landscape['Category'] = 'Landscape'
df_landscape_nfa = pd.read_csv("./rank_corr_results_means_landscape_nfa.csv",
index_col=False, 
header = 0
)
df_landscape_nfa['Category'] = 'Landscape'

# combine all dfs
df = pd.concat([df_all, df_sport, df_animal, df_vehicle, df_food, df_landscape])
df_nfa = pd.concat([df_all_nfa, df_sport_nfa, df_animal_nfa, df_vehicle_nfa, df_food_nfa, df_landscape_nfa])

# colours
bright=["#023EFF", "#FF7C00", "#1AC938", "#E8000B", "#8B2BE2",
            "#9F4800", "#F14CC1", "#A3A3A3", "#FFC400", "#00D7FF"]
col_dict = dict(Animal = bright[8], Sport = bright[0], Food = bright[4], Vehicle = bright[3], Landscape = bright[2], All=bright[7])

f, axs = plt.subplots(1,2,
                      figsize=(12,6),
                      sharey=True)

sns.set_theme(style="whitegrid", palette='bright', font_scale=1.5)

sns.scatterplot(data=df_nfa, x = 'nresp', y = 'corr', hue = 'Category', style='Category', palette=col_dict, alpha = 0.7, ax=axs[0], legend=False)
sns.scatterplot(data=df, x = 'nresp', y = 'corr', hue = 'Category', style='Category', palette=col_dict, alpha = 0.7, ax=axs[1])

axs[0].text(-0.1, 1.1, 'A', transform=axs[0].transAxes, size=20, weight='bold')
axs[1].text(-0.1, 1.1, 'B', transform=axs[1].transAxes, size=20, weight='bold')

axs[0].set(ylabel= r"Mean Split-Half Spearman's $\rho$", xlabel='Mean $n_{data \ points \ / \ image}$', title='H/N$_{resp}$')
axs[1].set(title='(H-F)/N$_{resp}$', ylabel='', xlabel='Mean $n_{data \ points \ / \ image}$')

axs[1].legend(loc='center left', bbox_to_anchor=(1, 0.5))

plt.savefig('./rank_corr_two.pdf', bbox_inches='tight')

