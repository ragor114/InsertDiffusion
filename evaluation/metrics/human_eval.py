import pandas as pd
import json
import numpy as np
from scipy.stats import f_oneway
from scipy.stats import ttest_rel
import pingouin as pg

import os
if os.getcwd().split('/')[-1] != 'evaluation' or os.getcwd().split('\\')[-1] != 'evaluation':
    os.chdir('..')

# This file contains functions to analyze results from a human evaluation study
# TODO: work in progress and currently not very flexible with respect to the input data (see below)

"""
The data for the functions here is already assumed to be preprocessed
Thus, in the dataframe the columns to be as follows (each row corresponds to one participant)
Optional: 'Timestamp'/'Zeitstempel'
Some demographic info e.g. 'Age', 'Gender', 'Education', ...
    Note that columnnames of demographic info may NOT include a _
Image Ratings: sorted by first mode, then method and then image with column headings of form '<metric>_<mode>_<method>_<dataset>_<id>'
    So for example GeneralAppeal_bg_Ours_data_1, GeneralAppeal_bg_Ours_data_2, ..., GeneralAppeal_bg_Other_data_1, GeneralAppeal_bg_Other_data_2, ...
Ratings should be of integer type
"""

# ----- Demographic data -----

def demographic_analysis(data: pd.DataFrame, save: bool=True, save_path: str='./images/human_eval/demographic.json') -> dict:
    columns = list(data.columns)
    demographic_columns = [x for x in columns if '_' not in x]
    try:
        demographic_columns.remove('Timestamp')
    except ValueError:
        pass
    try:
        demographic_columns.remove('timestamp')
    except ValueError:
        pass
    try:
        demographic_columns.remove('Zeitstempel')
    except ValueError:
        pass
    try:
        demographic_columns.remove('zeitstempel')
    except ValueError:
        pass
    df_demographic = data[demographic_columns]

    description = df_demographic.describe()
    remaining_columns = [x for x in df_demographic.columns if x not in description.columns]
    count_data = {c: df_demographic.groupby(c)[c].nuquie().to_dict() for c in remaining_columns}

    final = {}
    final['numeric'] = description.to_dict()
    final['categorical'] = count_data

    if save:
        os.makedirs('/'.join(save_path.split('/')[:-1]))
        with open(save_path, 'w') as f:
            json.dump(final, f)

    return final

# ----- Descriptive Statistics of Rating -----

def descriptive_statistics_for_mode(data: pd.DataFrame, mode: str, save_plot: bool=True, save_path: str='./images/human_eval/results/plots'):
    columns_mode = [x for x in data.columns if '_' + mode + '_' in x]
    df_mode = data[columns_mode]

    methods = set()
    metrics = set()
    datasets = set()
    for x in columns_mode:
        splits = x.split('_')
        metric = splits[0]
        method = splits[2]
        dataset = splits[3]
        metrics.add(metric)
        methods.add(method)
        datasets.add(dataset)
    methods = list(methods)
    metrics = list(metrics)
    datasets = list(datasets)

    results = {}
    for method in methods:
        for metric in metrics:
            relevant_columns = [x for x in df_mode.columns if method in x and metric in x]
            mean = np.mean(df_mode[relevant_columns].to_numpy())
            for dataset in datasets:
                c_columns = [x for x in relevant_columns if dataset in x]
                mean = np.mean(df_mode[c_columns].to_numpy())
                results[method + '_' + dataset + '_' + metric] = mean
            results[method + '_overall_' + metric] = mean

    metric_results = {metric: {x.replace('_'+ metric, ''): results[x] for x in results.keys() if metric in x} for metric in metrics}

    ds_results = {dataset+metric: {x.replace('_'+ dataset, ''): metric_results[metric][x] for x in metric_results[metric].keys() if dataset in x} for dataset in list(datasets)+['overall'] for metric in metrics}

    df_comp_results = pd.DataFrame.from_dict(ds_results)
    df_comp_results = df_comp_results.sort_index(axis=1)

    results_per_dataset = {}
    for dataset in datasets:
        columns = [x for x in df_comp_results.columns if dataset in x]
        curr = df_comp_results[columns]
        results_per_dataset[dataset] = curr.to_dict()

        rename_dict = {dataset+m: m for m in metrics}
        fig = curr.transpose().rename(rename_dict).plot.bar(title=dataset).get_figure()
        if save_plot:
            os.makedirs(save_path, exist_ok=True)
            fig.savefig(os.path.join(save_path, f'{mode}_{dataset}.png'))

    return results_per_dataset
    
def ratings_descriptive(data: pd.DataFrame, save: bool=True, save_path: str='./images/human_eval/results/descriptive.json', save_plot: bool=True, plot_save_path: str='./images/human_eval/results/plots'):
    columns = list(data.columns)
    rating_columns = [x for x in columns if '_' in x]
    df_ratings = data[rating_columns]
    
    modes = set()
    for col in rating_columns:
        modes.add(col.split('_')[1])
    modes = list(modes)
    results = {m: descriptive_statistics_for_mode(df_ratings, m, save_plot, plot_save_path) for m in modes}

    if save:
        os.makedirs('/'.join(save_path.split('/')[:-1]), exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(results)

    return results

# ----- Inferential Statistics of Ratings -----
# TODO: currently only repeated measures (within-subject) ANOVA/Friedmann test and posthoc t-tests with Bonferroni correction supported

def clean_rows(data: pd.DataFrame, methods: list):
    method_columns = [[x.split('_')[-1] for x in data.columns if m in x] for m in methods]
    sets = [set(x) for x in method_columns]
    union_set = set().union(*sets)
    intersection_set = set.intersection(*sets)
    not_in_all = list(union_set - intersection_set)
    columns_to_drop = [x for x in data.columns if x.split('_')[-1] in not_in_all]
    data = data.drop(columns=columns_to_drop)
    return data

def conduct_inferential_statistics(data: pd.DataFrame, methods: list, alpha: float=0.05):
    data = clean_rows(data, methods)
    data['pid'] = range(1, len(data) + 1)

    df_long = pd.wide_to_long(data, stubnames=[m+'_' for m in methods], i='pid', j='image')
    rename_dict = {x: x[:-1] for x in df_long.columns}
    df_long = df_long.rename(columns=rename_dict)
    df_long = df_long.reset_index()
    
    df_pg = data.melt(id_vars=['pid'], var_name='image', value_name='rating')
    df_pg[['method', 'image_index']] = df_pg['image'].str.split('_', expand=True)
    df_pg.drop(columns='image', inplace=True)
    df_pg = df_pg[['pid', 'image_index', 'method', 'rating']]
    normality_test = pg.normality(data=df_pg, group='method', dv='rating', method='shapiro')
    normality_test = normality_test.reset_index()
    normality = normality_test['normal'].all()

    if normality:
        aov = pg.rm_anova(data=df_pg, subject='pid', dv='rating', within='method', correction=True)
    else:
        aov = pg.friedman(data=df_pg, subject='pid', dv='rating', within='method', method='f')
        aov = aov.reset_index()

    p_anova = aov.loc[0, 'p-unc']
    sphericity = True
    if normality:
        if not aov.loc[0, 'sphericity']:
            sphericity = False
            p_anova = aov.loc[0, 'p-GG-corr']

    posthoc_results = []
    for i in range(len(methods)):
        for j in range(i+1, len(methods)):
            t_stat, p_val = ttest_rel(df_long[methods[i]], df_long[methods[j]])
            posthoc_results.append((methods[i], methods[j], t_stat, p_val))
    posthoc_corrected = [{'method1': m1, 'method2': m2, 't-statistic': t, 'corrected p': p * len(posthoc_results), 'significant?': bool(p *len(posthoc_results) < alpha)} for m1, m2, t, p in posthoc_results]

    return {
        'MainEffect': {
            'F': {
                'dof_numerator': aov.loc[0, "ddof1"],
                'dof_denominator': aov.loc[0, "ddof2"],
                'f-statistic': aov.loc[0, "F"]
            },
            'p': p_anova,
            'correction': 'Greenhouse-Geiser' if not sphericity else 'None',
            'test': 'ANOVA' if normality else 'Friedmann',
            'significant': bool(p_anova < alpha)
        },
        'PostHoc': posthoc_corrected
    }

# assumes column names 'metric_method_i'
def conduct_test_for_all_metrics(data: pd.DataFrame, methods: list, alpha: float=0.05):
    columns = list(data.columns)
    metrics = set()
    for c in columns:
        metrics.add(c.split('_')[0])
    metrics = list(metrics)
    results = {}
    for metric in metrics:
        relevant_columns = [x for x in columns if metric in x]
        df_curr = data[relevant_columns]
        rename_dict = {c: c.split('_')[1] + '_' + c.split('_')[-1] for c in relevant_columns}
        df_curr = df_curr.rename(columns=rename_dict)
        results[metric] = conduct_inferential_statistics(df_curr, methods, alpha)
    return results

# expects column names 'metric_method_dataset_i'
def conduct_test_for_dataset(data: pd.DataFrame, dataset: str, methods: list, alpha: float=0.05):
    columns = list(data.columns)
    relevant_columns = [c for c in columns if dataset in c]
    df_curr = data[relevant_columns]
    rename_dict = {c: c.split('_')[0] + '_' + c.split('_')[1] + '_' + c.split('_')[-1] for c in df_curr.columns}
    df_curr = df_curr.rename(columns=rename_dict)
    return conduct_test_for_all_metrics(df_curr, methods, alpha)

# expects column names 'metric_method_dataset_i'
def conduct_test_for_all_datasets(data: pd.DataFrame, methods: list, alpha: float=0.05):
    columns = list(data.columns)
    datasets = set()
    for c in columns:
        datasets.add(c.split('_')[2])
    datasets = list(datasets)
    results = {}
    for ds in datasets:
        results[ds] = conduct_test_for_dataset(data, ds, methods, alpha)
    return results
    
# expects column names 'metric_mode_method_dataset_i'
def conduct_test_for_modes(data: pd.DataFrame, alpha: float=0.05, save_path: str='./images/human_eval/results/inferential.json'):
    os.makedirs('/'.join(save_path.split('/')[-1]), exist_ok=True)
    columns = list(data.columns)
    modes = set()
    for c in columns:
        modes.add(c.split('_')[1])
    modes = list(modes)
    results = {}
    for mode in modes:
        relevant_columns = [x for x in columns if mode in x]
        df_curr = data[relevant_columns]
        rename_dict = {x: '_'.join([x.split('_')[0], x.split('_')[2], x.split('_')[3], x.split('_')[4]]) for x in df_curr.columns}
        df_curr = df_curr.rename(columns=rename_dict)

        methods = set()
        for c in df_curr.columns:
            methods.add(c.split('_')[1])
        methods = list(methods)
        print(f'mode: {mode} - methods: {methods}')

        results[mode] = conduct_test_for_all_datasets(df_curr, methods, alpha)

    if save_path is not None:
        with open(save_path, 'w') as f:
            json.dump(results, f)
    return results

def ratings_inferential(data: pd.DataFrame, sig_level: float=0.05, save: bool=True, save_path: str='./images/human_eval/results/inferential.json'):
    columns = list(data.columns)
    rating_columns = [x for x in columns if '_' in x]
    df_ratings = data[rating_columns]
    
    save_path = None if not save else save_path
    results = conduct_test_for_modes(df_ratings, sig_level, save_path)
    return results
