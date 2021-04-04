import streamlit as st
import altair as alt
import pandas as pd
import math
import numpy as np
import dill
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from ipywidgets import IntSlider
from ipywidgets import interact
from matplotlib import pylab as plt

df_counties_w_cluster_stats = dill.load(open('dataframes_saved/df_counties_w_cluster_stats', 'rb'))
df_intervdiff_clusters = dill.load(open('dataframes_saved/df_intervdiff_clusters.pkd', 'rb'))
df_posInc_policy = dill.load(open('dataframes_saved/df_posInc_policy', 'rb'))
df_all_SMA_7 = dill.load(open('dataframes_saved/df_all_SMA_7(4-1-2021).pkd','rb'))
df_state_fips = df_posInc_policy[['fips','stname']]
df_all_SMA_7 = df_all_SMA_7.merge(df_state_fips, on='fips')
df_prepan_pov_age_cluster = dill.load(open('dataframes_saved/df_prepan_pov_age_cluster.pkd','rb'))

df_cluster_pop_pov_age = df_prepan_pov_age_cluster[['Cluster', 'popdensity_2018','PCT_POV', 'PCT_AGED']]
df_cluster_stats = df_cluster_pop_pov_age.groupby(['Cluster']).mean() \
										.reset_index() \
										.rename(columns={'popdensity_2018': 'Pop. Density (2018)',
														'PCT_POV': '% in Poverty',
														'PCT_AGED': '% Seniors in Population'})
df_cluster_stats['% Seniors in Population'] = df_cluster_stats['% Seniors in Population'].apply(lambda x: x*100)
cluster_bar_chart = dill.load(open('filter_columns_bar_chart(4-3-2021).pkd', 'rb'))

# df_pan = pd.read_csv('medical.csv', low_memory=False)
#
# df_pan_cases = pd.read_csv('casesAndDeaths.csv')
# df_poverty = pd.read_csv('PovertyEstimates.csv')
# df_pop = pd.read_csv('PopulationEstimates.csv')

@st.cache()  #cache results of function calls; streamlit only runs once and saves result; loading data will be instant
def load_prepan():
	df_prepan = pd.read_csv('Prepandemic_v2.csv')
	columns = (['FIPS','stname','ctyname_x', 'tot_2018', 'popdensity_2018'])
	df_prepan = df_prepan[columns]
	return df_prepan

def make_summary_and_bar_chart(state, county):
	st.write('# Summary Report for {}: {}'.format(state, county))
	df_state_filter = df_counties_w_cluster_stats[df_counties_w_cluster_stats['stname'] == state]
	df_county_filter = df_state_filter[df_state_filter['ctyname'] == county]

	columns1 = ['pre_post_diff_Mask Mandate_x', 'pre_post_diff_Stay at home/ shelter in place_x', 'pre_post_diff_Restaurants Closed_x']
	columns2 = ['pre_post_diff_Mask Mandate_y', 'pre_post_diff_Stay at home/ shelter in place_y', 'pre_post_diff_Restaurants Closed_y']

	max_diff = df_county_filter[columns1].max(axis=1).item()
	cluster = df_county_filter['Cluster'].unique()[0]

	st.write('## Prediction')
	if cluster == '1':
		st.write('{}\'s {} is within Cluster 1. As such, our model predicts that Mask Mandates would be the most effective intervention in decreasing the spread of COVID-19, while the remaining interventions show no effect.'.format(state, county))
	elif cluster == '2':
		st.write('{}\'s {} is within Cluster 2. As such, our model predicts that none of the interventions will be effective in decreasing the spread of COVID-19.'.format(state, county))
	elif cluster == '3':
		st.write(('{}\'s {} is within Cluster 3. As such, our model predicts that would have a strong effect in decreasing the spread of COVID-19, while the remaining interventions show no effect.'.format(state, county)))


	st.write('## Observed Results')
	st.write('''
			While cluster labeling is being used as a predictor for intervention effects,
			some variance is expected as individual counties may exhibit outcomes that differ
			from their own cluster due to how they implement those measures.
	''')
	st.write(' The observed results for {} show that:'.format(county))
	if max_diff <= 0:
	    st.write('None of the policies reduced COVID-19 spread in {} State\'s {}.'.format(state, county))
	    for i in columns1:
	        split_i = i.split('_')
	        if max_diff == df_county_filter[i].item():
	            round_max_diff = round(max_diff, 3)
	            st.write('The {} policy was the most effective in slowing COVID-19 spread in {} State\'s {}.'.format(split_i[3], state, county))
	            break
	else:
	    for i in columns1:
	        split_i = i.split('_')
	        if max_diff == df_county_filter[i].item():
	            round_max_diff = round(max_diff, 3)
	            st.write('The {} policy is the most effective intervention in reducing COVID-19 spread in {} State\'s {}. It reduced the spread by {} per 1000 people after 3 weeks post intervention.'.format(split_i[3], state, county, round_max_diff))
	            break
	        if df_county_filter[i].item() != max_diff and df_county_filter[i].item() > 0:
	            round_item = round(df_county_filter[i].item(), 3)
	            st.write('The {} policy was also effective in reducing COVID-19 spread in {} State\'s {}. It reduced the spread by {} per 1000 people after 3 weeks post intervention'.format(split_i[3], state, county, round_item))
	#################################################################################################
	row_index_val = list(df_county_filter.index)[0]

	df_county_filter_short = df_county_filter[['pre_post_diff_Mask Mandate_x',
	                                       'pre_post_diff_Stay at home/ shelter in place_x',
	                                       'pre_post_diff_Restaurants Closed_x']]

	df_county_filter_short = df_county_filter_short.rename(columns={'pre_post_diff_Mask Mandate_x': 'Mask Mandate',
	                         'pre_post_diff_Stay at home/ shelter in place_x': 'Stay at home/ shelter in place',
	                         'pre_post_diff_Restaurants Closed_x':'Restaurants Closed'})

	df_county_filter_t = df_county_filter_short.transpose().reset_index().rename(columns={row_index_val: 'mean_diff', 'index': 'intervention'})
	df_county_filter_t['level'] = county
	############################################################
	df_cluster_filter_short = df_county_filter[['pre_post_diff_Mask Mandate_y',
	                                       'pre_post_diff_Stay at home/ shelter in place_y',
	                                       'pre_post_diff_Restaurants Closed_y']]

	df_cluster_filter_short = df_cluster_filter_short.rename(columns={'pre_post_diff_Mask Mandate_y': 'Mask Mandate',
	                         'pre_post_diff_Stay at home/ shelter in place_y': 'Stay at home/ shelter in place',
	                         'pre_post_diff_Restaurants Closed_y':'Restaurants Closed'})

	df_cluster_filter_t = df_cluster_filter_short.transpose().reset_index() \
	                                        .rename(columns={row_index_val: 'mean_diff', 'index': 'intervention'})
	df_cluster_filter_t['level'] = 'County\'s Cluster'
	############################################################
	df_intervdiff_clusters_renamed = df_intervdiff_clusters.rename(columns={'pre_post_diff_Mask Mandate': 'Mask Mandate',
	                                                                    'pre_post_diff_Stay at home/ shelter in place': 'Stay at home/ shelter in place',
	                                                                    'pre_post_diff_Restaurants Closed': 'Restaurants Closed'})
	df_meandiff_all = df_intervdiff_clusters_renamed[['Mask Mandate','Stay at home/ shelter in place','Restaurants Closed']]
	df_all_t = df_meandiff_all.mean().to_frame().reset_index().rename(columns = {0:'mean_diff', 'index': 'intervention'})
	df_all_t['level'] = 'U.S.A.'
	############################################################
	frames = [df_county_filter_t, df_cluster_filter_t, df_all_t]
	total_df_bar = pd.concat(frames)
	############################################################
	st.write('The following charts compare {}\'s performance to both its own cluster and to all U.S. counties'.format(county))
	bar_chart_compare = alt.Chart(total_df_bar).mark_bar().encode(
	    x='level:O',
	    y='mean_diff:Q',
	    color='level:N',
	    column='intervention:N',
	    tooltip=['mean_diff:Q']
	    ).properties(
	        height=500,
	        width=100
	    ).interactive()

	return bar_chart_compare
# May also want to return the print statements


def find_county_SMA(state, county, intervention):
    df_filtered = df_all_SMA_7[df_all_SMA_7['stname']==state]
    df_filtered = df_filtered[df_filtered['ctyname']==county]
    df_filtered = df_filtered[df_filtered['intervention']==intervention]
    df_county_SMA = df_filtered.groupby(['days since intervention']).mean().reset_index() \
                                .rename(columns={'SMA_7': '{}_SMA'.format(county)})
    return df_county_SMA
####################################################################################################
def find_cluster_SMA(cluster, intervention):
    df_filtered = df_all_SMA_7[df_all_SMA_7['cluster']==cluster]
    df_filtered = df_filtered[df_filtered['intervention']==intervention]
    df_cluster_SMA_7 = df_filtered.groupby(['days since intervention']).mean().reset_index() \
                                .rename(columns={'SMA_7': 'Cluster_{}_SMA'.format(cluster)})
    return df_cluster_SMA_7

####################################################################################################
def find_all_SMA(intervention):
    df_filtered = df_all_SMA_7[df_all_SMA_7['intervention']==intervention]
    df_all_SMA = df_filtered.groupby(['days since intervention']).mean().reset_index() \
                                .rename(columns={'SMA_7': 'all_SMA'})
    return df_all_SMA
####################################################################################################
def get_cluster_label(state, county):
    #county = county + " County"
    df_temp = df_all_SMA_7[df_all_SMA_7['stname']==state]
    df_temp = df_temp[df_temp['ctyname']==county]
    cluster = df_temp.iloc[0]['cluster']
    return cluster
###################################################################################################
##################################################################################################
def merge_multiline_per_interv(state, county, intervention):
		cluster = get_cluster_label(state, county)
		df_county_SMA = find_county_SMA(state, county, intervention)
		df_cluster_SMA = find_cluster_SMA(cluster, intervention)
		df_all_SMA = find_all_SMA(intervention)
		df_merged = df_county_SMA.merge(df_cluster_SMA, on = 'days since intervention') \
                                .merge(df_all_SMA, on = 'days since intervention')
		df_merged['intervention'] = intervention
		return df_merged
############################################
def create_multiline(state, county, ls_interventions):
    frames = []
    for interv in ls_interventions:
        df = merge_multiline_per_interv(state, county, interv)
        frames.append(df)
    df_concat_all = pd.concat(frames)
    cluster = get_cluster_label(state, county)
    df_long = df_concat_all.melt(id_vars=['days since intervention','intervention'], value_vars=['{}_SMA'.format(county), 'Cluster_{}_SMA'.format(cluster), 'all_SMA'])

    base = alt.Chart(df_long, title = "SMA ({} vs. Cluster {} vs. All)".format(county, cluster)).mark_line(point=True).encode(
    x='days since intervention',
    y=alt.Y('value', title='SMA (New Cases per 1000 People)'),
    color='variable',
    tooltip=['days since intervention:Q', 'value:Q']
    ).properties(
        height=375,
        width=500
    ).interactive()

    # A dropdown filter
    interventions = ['Mask Mandate', 'Stay at home/ shelter in place', 'Restaurants Closed']
    column_dropdown = alt.binding_select(options=interventions)
    column_select = alt.selection_single(
        fields=['intervention'],
        on='doubleclick',
        clear=False,
        bind=column_dropdown,
        name='Select',
        init={'intervention': 'Mask Mandate'}
    )

    filter_columns = base.add_selection(
        column_select
    ).transform_filter(
        column_select
    )

    return filter_columns.interactive()


def app():
	st.title("Targeted Strategies in Reducing the Spread of COVID-19 and Future Pandemics")
	# And write the rest of the app inside this function!
	st.sidebar.header("Header 1")
	st.markdown('''This web application is intended to assist in identifying effective policies to combat a respiratory-spread pandemic based on county-specific features.
				The model is based on county-level disease data collected during the COVID-19 pandemic, combined with census data for county-level demographics.
				''')
	st.markdown('''## Methodology''')
	st.markdown('''We will use 'k-means clustering' to identify 3 (three) distinct groups of U.S. counties based on the following characteristics:''')
	st.markdown('''
	- Population Density
	- Poverty Rate
	- Percent of Seniors in Population (Ages 65+)
				''')
	st.markdown('''The summary characteristics of each group are presented below: ''')
## Include df of cluster summary stats.
	st.write(df_cluster_stats)
	st.markdown('''We will then utilize each county's cluster label to predict which of the following
				interventions is most effective in decreasing or slowing the spread of COVID-19 and future pandemics:
				''')
	st.markdown('''
	- Mask Mandate
	- Stay at Home/ Shelter in Place
	- Non-essential Business Closures
				''')

	st.markdown('Our outcome measure will be based on the number of new daily cases per 1000 people in each county. We will calculate the 7-day moving average of this measure at two time points: the date when an intervention was implemented, and again 3 weeks after date of implementation. The final outcome measure will be the difference between these two points, where a positive value indicates a decrease in new cases per 1000 people (i.e. intervention has a perceived effect), while a negative value indicates an increase in new cases (i.e. intervention does not have a perceived effect).')
	st.markdown('The following bar chart depicts the average value of our outcome variable for each cluster and intervention. Use the drop down menu to observe the effects from different interventions.')

	## Insert bar chart!
	st.write(cluster_bar_chart)
	st.markdown('''Let's get started: ''')
	states_tup = tuple(list(df_counties_w_cluster_stats['stname'].unique()))
	state = st.selectbox('Select your state',
				states_tup)
	df_state_filter = df_counties_w_cluster_stats[df_counties_w_cluster_stats['stname'] == state]
	counties_tup = tuple(list(df_state_filter['ctyname']))
	county = st.selectbox('Select your county', counties_tup)

	interv_ls = list(df_all_SMA_7['intervention'].unique())

	if st.button('Run Report'):
		st.write(make_summary_and_bar_chart(state, county))
		st.write(create_multiline(state, county, interv_ls))
		# intervention = st.selectbox('Select intervention for multi-line graph', interv_tup)
		# if st.button('Run Multi-line Graph'):
		# 	st.write(multi_line_graph(state, county, intervention))

	# df_prepan = load_prepan()
	#
	# st.write(df_prepan)

	#st.sidebar.markdown('''Ahh yeah...bullet points!''')

if __name__ == '__main__':
	app()
