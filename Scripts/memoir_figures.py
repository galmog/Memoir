"""
@Author: Gal Almog
Code for Memoir figures
"""
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, spearmanr
import seaborn as sns 
import matplotlib.pyplot as plt
from collections import Counter

df = pd.read_csv("./memoir_results.csv",
index_col=False, 
usecols=['image_file','category','subcategory','FA', 'n_resp','memorability_wo_fa_correction','memoir_f', 'memoir_nresp', 'memoir_mem_score_yfa','memoir_mem_score_nfa', 'mean_response_time'],
header = 0
)

df.columns= ['image_file','category','subcategory', 'memcat_fa', 'memcat_nresp','memcat','memoir_fa', 'memoir_nresp','memoir_yfa','memoir_nfa','response_time']
# set the index to be the participant ids
df.set_index(keys=['image_file'], drop=True,inplace=True)
df['category'] = df['category'].str.title()

#sub dfs for each category
df_animal = df.loc[df['category'] == 'Animal']
df_sports = df.loc[df['category'] == 'Sports']
df_food = df.loc[df['category'] == 'Food']
df_vehicle = df.loc[df['category'] == 'Vehicle']
df_landscape = df.loc[df['category'] == 'Landscape']

# colours
bright=["#023EFF", "#FF7C00", "#1AC938", "#E8000B", "#8B2BE2","#9F4800", "#F14CC1", "#A3A3A3", "#FFC400", "#00D7FF"]
col_dict = dict(Animal = bright[8], Sports = bright[0], Food = bright[4], Vehicle = bright[3], Landscape = bright[2], All=bright[7])

"""
Violin Plots
"""
f, axs = plt.subplots(1,2,
                      figsize=(12,6),
                      sharey=True)

sns.set_theme(style="whitegrid", palette='bright', font_scale=1.1)

sns.violinplot(x="category", y="memoir_nfa", data=df, ax=axs[0], palette= col_dict, alpha=1, cut=0, order=['Food', 'Animal','Sports','Vehicle','Landscape'])
sns.violinplot(x="category", y="memcat", data=df, ax=axs[1], palette= col_dict, alpha=1, cut=0, order=['Food', 'Animal','Sports','Vehicle','Landscape'])
axs[0].axhline(df["memoir_nfa"].mean(), ls='--', c='red') #add mean line
axs[0].text(-0.1, 1.1, 'A', transform=axs[0].transAxes, size=20, weight='bold')
axs[1].axhline(df["memcat"].mean(), ls='--', c='red')
axs[1].text(-0.1, 1.1, 'B', transform=axs[1].transAxes, size=20, weight='bold')
axs[0].set(ylabel= 'Memorability Score', xlabel='', title='Memoir H/N$_{resp}$')
# axs[0].set(ylabel= 'Memorability Score', xlabel='', title='H/N$_{resp}$')
axs[1].set(title='MemCat H/N$_{resp}$', ylabel='', xlabel='')
# axs[1].set(title='(H-F)/N$_{resp}$', ylabel='', xlabel='')
plt.savefig('./memoir_dist_nfa.pdf', bbox_inches='tight')

"""
Matched comparison of Memoir and MeMcat
"""
f, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2,3,figsize=(12,6), sharey=True, sharex=True)
xlims=(0,1)
ylims=(0,1)

#1 = global
sns.scatterplot(ax = ax1, data=df, x = 'memoir_nfa', y = 'memcat', hue = 'category', palette= col_dict, legend=False, alpha=0.7)
# axs[0,0].set(xlabel='Adolescent Memorability Score', ylabel= 'Adult Memorability Score')
# axs[0,0].legend(loc='upper right', bbox_to_anchor=(1.4, 0.6), title='Category')
ax1.plot(xlims, ylims,c='black')
ax1.set_title(r'All Categories, $\rho$ = {}'.format(round(((spearmanr(df['memcat'], df['memoir_nfa']))[0]), 2)))
ax1.text(-0.1, 1.2, 'A', transform=ax1.transAxes, size=20, weight='bold')

# 2 - animal
sns.scatterplot(ax = ax2, data=df_animal, x = 'memoir_nfa', y = 'memcat', color = col_dict['Animal'], legend=False)
ax2.plot(xlims, ylims,c='black')
ax2.set_title(r'Animal, $\rho$ = {}'.format(round(((spearmanr(df_animal['memcat'], df_animal['memoir_nfa']))[0]), 2)))
ax2.text(-0.1, 1.2, 'B', transform=ax2.transAxes, size=20, weight='bold')

# 3 - sports
sns.scatterplot(ax = ax3, data=df_sports, x = 'memoir_nfa', y = 'memcat', color = col_dict['Sports'], legend=False)
ax3.plot(xlims, ylims,c='black')
ax3.set_title(r'Sport, $\rho$ = {}'.format(round(((spearmanr(df_sports['memcat'], df_sports['memoir_nfa']))[0]), 2)))
ax3.text(-0.1, 1.2, 'C', transform=ax3.transAxes, size=20, weight='bold')

# 4 - food
sns.scatterplot(ax = ax4, data=df_food, x = 'memoir_nfa', y = 'memcat', color = col_dict['Food'], legend=False)
ax4.plot(xlims, ylims,c='black')
ax4.set_title(r'Food, $\rho$ = {}'.format(round(((spearmanr(df_food['memcat'], df_food['memoir_nfa']))[0]), 2)))
ax4.text(-0.1, 1.2, 'D', transform=ax4.transAxes, size=20, weight='bold')

# 5 - vehicle
sns.scatterplot(ax = ax5, data=df_vehicle, x = 'memoir_nfa', y = 'memcat', color = col_dict['Vehicle'], legend=False)
ax5.plot(xlims, ylims,c='black')
ax5.set_title(r'Vehicle, $\rho$ = {}'.format(round(((spearmanr(df_vehicle['memcat'], df_vehicle['memoir_nfa']))[0]),2)))
ax5.text(-0.1, 1.2, 'E', transform=ax5.transAxes, size=20, weight='bold')

# 6 - landscape
sns.scatterplot(ax = ax6, data=df_landscape, x = 'memoir_nfa', y = 'memcat', color = col_dict['Landscape'], legend=False)
ax6.plot(xlims, ylims,c='black')
ax6.set_title(r'Landscape, $\rho$ = {}'.format(round(((spearmanr(df_landscape['memcat'], df_landscape['memoir_nfa']))[0]), 2)))
ax6.text(-0.1, 1.2, 'F', transform=ax6.transAxes, size=20, weight='bold')

for ax in ax1, ax2, ax3, ax4, ax5, ax6:
    ax.set(xlabel='Adolescent Memorability Score', ylabel='Adult Memorability Score')
    ax.label_outer()

plt.subplots_adjust(left=0.1,
                    bottom=0.1, 
                    right=0.9, 
                    top=0.9, 
                    wspace=0.4, 
                    hspace=0.4)

plt.savefig('./dist_scatter_multi_nfa.pdf', bbox_inches='tight')

"""
FA/HR Analysis
"""

df['memcat_fa_ratio'] = (df['memcat_fa']/df['memcat_nresp']) 
df['memoir_fa_ratio'] = (df['memoir_fa']/df['memoir_nresp'])

df['fa_rank'] = df['memoir_fa_ratio'].rank()
df['mem_rank'] = df['memoir_nfa'].rank()
df['fa_rank_memcat'] = df['memcat_fa_ratio'].rank()
df['mem_rank_memcat'] = df['memcat'].rank()

ttest_rel(df['memcat_fa_ratio'].values, df['memoir_fa_ratio'].values, alternative='less', axis = 0)

df_q1 = df[((df['fa_rank'] > np.median(df['fa_rank'])) & (df['mem_rank'] > np.median(df['mem_rank'])))]
df_q1_memcat = df[((df['fa_rank_memcat'] > np.mean(df['fa_rank_memcat'])) & (df['mem_rank_memcat'] > np.median(df['mem_rank_memcat'])))]
# spearmanr(df_q1['memcat'], df_q1['memoir_yfa'])

df_q2 = df[((df['fa_rank'] > np.median(df['fa_rank'])) & (df['mem_rank'] < np.median(df['mem_rank'])))]
df_q2_memcat = df[((df['fa_rank_memcat'] > np.mean(df['fa_rank_memcat'])) & (df['mem_rank_memcat'] < np.median(df['mem_rank_memcat'])))]
# spearmanr(df_q2['memcat'], df_q2['memoir_yfa'])

df_q3 = df[((df['fa_rank'] < np.median(df['fa_rank'])) & (df['mem_rank'] < np.median(df['mem_rank'])))]
df_q3_memcat = df[((df['fa_rank_memcat'] < np.mean(df['fa_rank_memcat'])) & (df['mem_rank_memcat'] < np.median(df['mem_rank_memcat'])))]
# spearmanr(df_q3['memcat'], df_q3['memoir_yfa'])

df_q4 = df[((df['fa_rank'] < np.median(df['fa_rank'])) & (df['mem_rank'] > np.median(df['mem_rank'])))]
df_q4_memcat = df[((df['fa_rank_memcat'] < np.mean(df['fa_rank_memcat'])) & (df['mem_rank_memcat'] > np.median(df['mem_rank_memcat'])))]
# spearmanr(df_q4['memcat'], df_q4['memoir_yfa'])

quadrants_counts = pd.DataFrame({
    'q1':pd.Series(Counter(df_q1_memcat.category)),
    'q2':pd.Series(Counter(df_q2_memcat.category)),
    'q3':pd.Series(Counter(df_q3_memcat.category)),
    'q4':pd.Series(Counter(df_q4_memcat.category)),
    })
quadrants_counts = pd.melt(quadrants_counts.reset_index(), id_vars='index',value_vars=['q1','q2','q3','q4'])

plt.figure(figsize=(12,8))
sns.scatterplot(data=df, x= 'memoir_fa_ratio', y='memoir_nfa', hue='category', style = 'category',alpha=0.7, palette=col_dict)

plt.xlabel('False Alarm Rate')
plt.ylabel('Hit Rate')
          
plt.axhline(y=df.memoir_nfa.median(), color='k', linestyle='--', linewidth=2)           
plt.axvline(x=df.memoir_fa_ratio.median(), color='k',linestyle='--', linewidth=2)
plt.legend(title='Category')
plt.savefig('./false_alarms_nfa.pdf', bbox_inches='tight')

plt.show()