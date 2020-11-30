import json

import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def _ingest_json():
	resp = requests.get('https://pomber.github.io/covid19/timeseries.json')
	resp_json_str = resp.text
	resp_dict = json.loads(resp_json_str)

	l = []
	for c, cl in resp_dict.items():
		for d in cl:
			d.update({'country' : c})
			l.append(d)
	df = pd.DataFrame(l)

	return df

def load_single_country(country='') -> pd.DataFrame:
	## Ingest JSON endpoint
	df_universe = _ingest_json()
	df_c = df_universe[df_universe.country==country].copy()

	## Preprocessing
	df_c['date'] = pd.to_datetime(df_c['date'], format='%Y-%m-%d')
	df_c['date'] = df_c['date'].dt.strftime('%Y-%m-%d')
	df_c['date'] = df_c['date'].astype(str)

	## Feature Engineering
	# Feature: New cases
	df_c['confirmed_tm1'] = df_c['confirmed'].shift(1)
	df_c['confirmed_tm1'] = df_c['confirmed_tm1'].fillna(0)
	df_c['confirmed_tm1'] = df_c['confirmed_tm1'].astype(int)
	df_c['new'] = df_c['confirmed'] - df_c['confirmed_tm1']
	_ = df_c.drop(['confirmed_tm1'], axis=1, inplace=True)
	# Feature: Currently Hospitalised
	df_c['active'] = df_c['confirmed'] - df_c['recovered'] - df_c['deaths']

	## Output
	_ = df_c.rename(columns={'confirmed' : 'total',
	'recovered' : 'discharged'}, inplace=True)
	_ = df_c.reset_index(drop=True, inplace=True)
	# Follow Mothership's columns naming
	c_cols = ['country', 'date',
	'total', 'new', 'deaths', 'discharged', 'active']
	return df_c[c_cols]

# Ingest to df
CTRY = 'Singapore'
df_sg = load_single_country(country=CTRY)

# Date Display
df_sg['date_dt'] = pd.to_datetime(df_sg['date'])
df_sg['date_mm'] = df_sg['date_dt'].dt.strftime('%m')
df_sg['date_dd'] = df_sg['date_dt'].dt.strftime('%d')
df_sg['date_mmdd'] = df_sg['date_dt'].dt.strftime('%m-%d')

# t_s100: Days after 100 cases have surpassed
# t_s100 = 0 when no. of cases exceed 100, -1 before this date
t_100cases = df_sg[df_sg.total>100].index.min()
df_sg['t'] = df_sg.index
df_sg['t_s100'] = df_sg['t'].apply(lambda x: x-t_100cases)
df_sg['t_s100'] = df_sg['t_s100'].apply(lambda x: x if x>=0 else -1)

df_sg['total_pct_change'] = df_sg['total'].pct_change()
df_sg['total_pct_change'] = df_sg['total_pct_change'].fillna(0.0)
df_sg['total_pct_change'] = df_sg['total_pct_change'].replace(float('inf'), 0.0)
# total_pct_change_rolling_mean: rolling mean (7 lags) of pct_change in cases
df_sg['total_pct_change_rolling_mean'] = df_sg['total_pct_change'].rolling(7).mean()
df_sg['total_pct_change_rolling_mean'] = df_sg['total_pct_change_rolling_mean'].fillna(0.0)

# Settings
NEW__COLNAME = 'new'
CONFIRMED__COLNAME, CONFIRMED__COLOR = 'total', 'FIREBRICK'
ACTIVE__COLNAME, ACTIVE__COLOR = 'active', 'DARKGREEN'
DISCHARGED__COLNAME, DISCHARGED__COLOR  = 'discharged', 'NAVY'
DEATHS__COLNAME, DEATHS__COLOR  = 'deaths', 'DARKSLATEGRAY'

LAST_DAYS = 30

# Constants
DATE__FIRSTCASE = '2020-01-23'
DATE__DORSCONORANGE = '2020-02-07'
DATE__CIRCUITBREAKER1 = '2020-04-07'
DATE__CIRCUITBREAKER2 = '2020-04-21'
DATE__CIRCUITBREAKER3 = '2020-05-12'
DATE__PHASE1 = '2020-06-02'
DATE__PHASE2 = '2020-06-19'

# Derived Values
M = df_sg['total_pct_change_rolling_mean'].mean()
count_confirmed = df_sg.tail(1)[CONFIRMED__COLNAME].tolist()[0]
count_active = df_sg.tail(1)[ACTIVE__COLNAME].tolist()[0]
count_discharged = df_sg.tail(1)[DISCHARGED__COLNAME].tolist()[0]
count_deaths = df_sg.tail(1)[DEATHS__COLNAME].tolist()[0]

# Notes
annotate_kwargs = dict(
    s='Based on COVID Data Repository by Johns Hopkins CSSE \nNews updates by CNA, Straits Times\nJavier Tan',
    xy=(0.1, 0.1), xycoords='figure fraction', fontsize=10)

note1text = "2020-01-23: First case detected"
note2text = "2020-02-07: Dorscon Orange Declared"
note2atext = "OTHERS\n2020-03-27: Bars, cinemas and all other entertainment outlets closed, \ntuition & enrichment classes suspended, all religious services suspended"
note2btext = "2020-05-28: MOH revises discharge criteria for Covid-19 patients, \nthose who are well by day 21 can be discharged"
note3text = "2020-04-07: Circuit Breaker Begins"
note4text = "2020-04-21: Circuit Breaker Tighter Measures Start"
note5text = "2020-05-12: Circuit Breaker Measures Relaxed"
note6text = "2020-06-02: Phase 1 Begins"
note7text = "2020-06-19: Phase 2 Begins"

# Generate values for date ranges
x = np.arange(df_sg.shape[0])
df2 = pd.DataFrame({'x' : x, 'y' : 0})

month_dates = ['2020-02-01', '2020-03-01', '2020-04-01', '2020-05-01', '2020-06-01',
               '2020-07-01', '2020-08-01', '2020-09-01', '2020-10-01', '2020-11-01']
month_names = ["FEB", "MAR", "APR", "MAY", "JUN",
               "JUL", "AUG", "SEP", "OCT", "NOV"]
month_indices = []
for md in month_dates:
    month_indices.append(df_sg[df_sg.date==md].index[0])

fig = plt.figure(figsize=(16,8))
ax = fig.add_subplot(1,1,1)

# Calculations before plotting
ylim_max = df_sg[CONFIRMED__COLNAME].max() * 1.2

# Plot SG statistics
df_sg.plot(kind='line', x='date', y=CONFIRMED__COLNAME, ax=ax, color=CONFIRMED__COLOR)
df_sg.plot(kind='line', x='date', y=ACTIVE__COLNAME, ax=ax, color=ACTIVE__COLOR)
df_sg.plot(kind='line', x='date', y=DISCHARGED__COLNAME, ax=ax, color=DISCHARGED__COLOR)
df_sg.plot(kind='line', x='date', y=DEATHS__COLNAME, ax=ax, color=DEATHS__COLOR)

# Axes
ax.set_xlabel("Date", fontsize=14)
ax.set_xticks(range(0,df_sg.shape[0],7))
ax.set_xticklabels(df_sg['date_mmdd'].iloc[::7],rotation=90)
ax.set_ylim(0, ylim_max)
ax.set_ylabel("No. of Cases", fontsize=14)

ax.fill_between([0,x.max()], ylim_max*0.0, ylim_max*0.1, facecolor='SILVER')

for md, (m1,m2) in enumerate(zip(month_dates, month_indices)):
    if md % 2 == 0:
        ax.fill_between([m2,x.max()], ylim_max*0.0, ylim_max*0.1, facecolor='WHITESMOKE')
    else:
        ax.fill_between([m2,x.max()], ylim_max*0.0, ylim_max*0.1, facecolor='SILVER')

# First Case
fc_index = df_sg[df_sg.date==DATE__FIRSTCASE].index[0]
ax.fill_between([fc_index,x.max()], ylim_max*0.1, ylim_max, facecolor='PEACHPUFF')
# DORSCON Orange
do_index = df_sg[df_sg.date==DATE__DORSCONORANGE].index[0]
ax.fill_between([do_index,x.max()], ylim_max*0.1, ylim_max, facecolor='PAPAYAWHIP')
# Circuit Breaker
cb1_index = df_sg[df_sg.date==DATE__CIRCUITBREAKER1].index[0]
ax.fill_between([cb1_index,x.max()], ylim_max*0.1, ylim_max, facecolor='MOCCASIN')
# Circuit Breaker Tighter Measures
cb2_index = df_sg[df_sg.date==DATE__CIRCUITBREAKER2].index[0]
ax.fill_between([cb2_index,x.max()], ylim_max*0.1, ylim_max, facecolor='PEACHPUFF')
# Circuit Breaker
cb3_index = df_sg[df_sg.date==DATE__CIRCUITBREAKER3].index[0]
ax.fill_between([cb3_index,x.max()], ylim_max*0.1, ylim_max, facecolor='MOCCASIN')
# Phase 1
p1_index = df_sg[df_sg.date==DATE__PHASE1].index[0]
ax.fill_between([p1_index,x.max()], ylim_max*0.1, ylim_max, facecolor='THISTLE')
# Phase 2
p2_index = df_sg[df_sg.date==DATE__PHASE2].index[0]
ax.fill_between([p2_index,x.max()], ylim_max*0.1, ylim_max, facecolor='LIGHTGREEN')

# Grid & Legend
ax.grid(color='LIGHTGRAY')
ax.legend([CONFIRMED__COLNAME, ACTIVE__COLNAME, DISCHARGED__COLNAME, DEATHS__COLNAME], fontsize=12, loc=6)

# Text
ax.set_title("No. of cases in {}, daily".format(CTRY), fontsize=16)

ax.text(df_sg.index.max()-2, count_confirmed+1000, str(count_confirmed), fontsize=14, fontweight='bold', color=CONFIRMED__COLOR)
ax.text(df_sg.index.max()-2, count_active+3500, str(count_active), fontsize=14, fontweight='bold', color=ACTIVE__COLOR)
ax.text(df_sg.index.max()-2, count_discharged-3000, str(count_discharged), fontsize=14, fontweight='bold', color=DISCHARGED__COLOR)
ax.text(df_sg.index.max()-2, count_deaths+500, str(count_deaths), fontsize=14, fontweight='bold', color=DEATHS__COLOR)

tbegin, tstep, tfsize = 0.98, 0.05, 10
tcolor='BROWN'
ax.text(fc_index+0.5, ylim_max*(tbegin-tstep*1),note1text, fontsize=tfsize, color=tcolor)
ax.text(do_index+0.5, ylim_max*(tbegin-tstep*2),note2text, fontsize=tfsize, color=tcolor)
ax.text(cb1_index+0.5, ylim_max*(tbegin-tstep*3),note3text, fontsize=tfsize, color=tcolor)
ax.text(cb2_index+0.5, ylim_max*(tbegin-tstep*4),note4text, fontsize=tfsize, color=tcolor)
ax.text(cb3_index+0.5, ylim_max*(tbegin-tstep*5),note5text, fontsize=tfsize, color=tcolor)
ax.text(p1_index+0.5, ylim_max*(tbegin-tstep*6),note6text, fontsize=tfsize, color=tcolor)
ax.text(p2_index+0.5, ylim_max*(tbegin-tstep*7),note7text, fontsize=tfsize, color=tcolor)

# ax.text(do_index+0.5, ylim_max*(tbegin-tstep*(7+3)),note2atext, fontsize=10, color=tcolor)
# ax.text(do_index+0.5, ylim_max*(tbegin-tstep*(7+4.5)),note2btext, fontsize=10, color=tcolor)

# Months
for md, (m2, m3) in enumerate(zip(month_indices, month_names)):
    ax.text(m2+0.5,ylim_max*0.07,m3, fontsize=12, color="BLACK")

# Annotations
ax.annotate(**annotate_kwargs)
plt.subplots_adjust(bottom=0.25)
plt.show()
