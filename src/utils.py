import os
import gc
import re
import requests
from urllib.parse import urlparse
import pyreadstat
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from scipy.stats import spearmanr, norm

from src.directory import data_dir, NHANES_dir, NHANES_vars_lookup_filename
from src.data_dict import NHANES_transformations, binary_response_dict
from src.data_dict import *


def get_descriptive_stats(df, numerical_features):
    descriptive_stats = df[numerical_features].describe().loc[['mean', 'std']]
    return descriptive_stats.to_dict()


def test_unconfoundedness_by_feature(df, outcome_col, conditional_col, feature_cols, alpha=0.05):
    confounders = pd.DataFrame(index=feature_cols, 
                               columns=['corr',
                                        'p_value'])
    confounders.index.name = 'feature'

    Y = df[outcome_col].to_numpy()
    T = df[conditional_col].to_numpy()
    
    for feature in feature_cols:
        # split data
        X = df[[feature]].to_numpy()
        X_train, X_test, T_train, T_test, Y_train, Y_test = train_test_split(X, T, Y, test_size=0.2, random_state=40)

        # standardize X
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # fit model 
        model = LogisticRegression()
        model.fit(X_train, T_train) # fit model
        
        # get propensity scores
        propensity_scores = model.predict_proba(X_test)[:, 1]

        # calculate correlation between outcome and propensity
        corr = spearmanr(Y_test, propensity_scores)

        # # get propensity scores
        # propensity_scores = get_propensity_scores(df, T_col=conditional_col, X_cols=[feature])

        # corr = spearmanr(Y, propensity_scores)

        confounders.loc[feature] = corr.statistic, corr.pvalue
    
    # determine confound by significance testing
    confounders.loc[:, 'confounder'] = confounders.loc[:, 'p_value'] < alpha

    return confounders


def get_propensity_scores(df, T_col, X_cols):
    T = df[T_col].to_numpy()
    
    # split data
    X = df[X_cols].to_numpy()

    # standardize X
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # fit model 
    model = LogisticRegression()
    model.fit(X, T) # fit model
    
    # get propensity score
    propensity_score = model.predict_proba(X)[:, 1]

    return propensity_score

def one_hot(df, cols):
    """
    @param df pandas DataFrame
    @param cols a list of columns to encode 
    @return a DataFrame with one-hot encoding
    """
    for col in cols:
        dummies = pd.get_dummies(df[col], prefix=col, drop_first=False)
        df = pd.concat([df, dummies], axis=1)
        del df[col]
    return df

def check_estimator_normality(value, variance, alpha=0.05):
    std_dev = np.sqrt(variance)
    z_score = value / std_dev

    p_value = 2 * (1 - norm.cdf(abs(z_score)))

    return {z_score:z_score, p_value:p_value, 'significant': p_value < alpha}

def check_strings_in_columns(df, columns, strings):
    results = {}
    for index, row in df.iterrows():
        found = False
        for col in columns:
            if isinstance(row[col], str):  # Ensure we're dealing with strings
                for s in strings:
                    if s in row[col]:
                        found = True
                        break  # No need to check other strings or columns for this row
            if found:
                break
        results[index] = 1 if found else 0
    return results

def read_sas_chunk(filename, chunksize=5e5, offset=0, columns=None):
        
    # Get the function object in a variable getChunk
    if filename.lower().endswith('sas7bdat'):
        getChunk = pyreadstat.read_sas7bdat
    else:
        getChunk = pyreadstat.read_xport
        
    offset += chunksize
    chunk, _ = getChunk(filename, row_limit=chunksize, row_offset=offset, usecols=columns)
        
    return chunk

def get_NHANES_questionnaire_df(vars_lookup_df, dataset_path, columns, index='SEQN', q_name='Questionnaire'):
    # read in dataframe
    encoding = "utf-8" if 'RXQ_RX_H.xpt' not in dataset_path else None
    
    if 'PAXMIN_H.xpt' in dataset_path: # ambient light
        offset = 0
        chunk_size = 1e6
        
        def get_nightly_lux(df):
            sleep_wear = df[df['PAXPREDM']==2]  # predicted sleep status
            total_sleep_minutes = df['PAXPREDM'].groupby(index).count() # total sleep minutes
            summed_lux_asleep = sleep_wear['PAXLXMM'].groupby(index).sum() # melanopic EDI
            df = pd.concat([total_sleep_minutes, summed_lux_asleep], axis=1).rename(columns={'PAXPREDM':'total_sleep_minutes', 'PAXLXMM':'summed_lux'})
            return df
        
        df = read_sas_chunk(dataset_path, chunk_size, offset, [index, *columns]).set_index(index).astype(float)
        df = get_nightly_lux(df)
        
        empty_chunk = False
        while not empty_chunk:
            offset += chunk_size
            chunk = read_sas_chunk(dataset_path, chunk_size, offset, [index, *columns]).set_index(index).astype(float)
            print(f'Processing chunk {offset} - {offset + chunk_size}')
            empty_chunk = chunk.empty 
            chunk = get_nightly_lux(chunk)
            
            df = pd.concat([df, chunk]).groupby(index).sum()

            del chunk
            gc.collect()
        
        # get nightly lux per minute
        df[light_col] = df['summed_lux'] / df['total_sleep_minutes']
    
    else:
        df = pd.read_sas(dataset_path, index=index, encoding=encoding)[columns]
    
    if 'PAQ_H.xpt' in dataset_path: # physical activity self-reported
        vigorous_moderate_activity_cols = NHANES_transformations[physical_activity_col]
        
        # get indicator variable
        df[vigorous_moderate_activity_cols] = df[vigorous_moderate_activity_cols].replace(binary_response_dict)
        df[physical_activity_col] = df[vigorous_moderate_activity_cols].sum(axis=1, min_count=1)
        df[physical_activity_col] = df[physical_activity_col].apply(lambda x: 1 if x > 0 else x)
        
        df[sedentary_col] = df['PAD680'].replace({7777:float('nan'), 
                                                  9999:float('nan')}) / (24 * 60) # percentage of day sedentary
        
        # drop unnecessary columns
        df.drop(vigorous_moderate_activity_cols + 
                NHANES_transformations[sedentary_col], axis=1, inplace=True)
    
    if 'PAXDAY_H.xpt' in dataset_path: # day sum of accelerometer triaxial values
        df = df.groupby(index).mean().rename(
            columns={NHANES_transformations[accelerometer_col][0]:accelerometer_col})
    
    if 'BPX_H.xpt' in dataset_path: # blood pressure examination
        valid_BP = df['PEASCST1'].replace({2:1, 3:0, '':float('nan')}) # partial, not done, missing
        systolic_cols = ['BPXSY1','BPXSY2','BPXSY3','BPXSY4']
        diastolic_cols = ['BPXDI1','BPXDI2','BPXDI3','BPXDI4']
        
        diastolic_BP = df[diastolic_cols].mean(axis=1)
        df[diastolic_col] = diastolic_BP
        high_diastolic_BP = diastolic_BP.apply(lambda x: 0 if x < DBP_cutoff else x)
        high_diastolic_BP = high_diastolic_BP.apply(lambda x: 1 if x >= DBP_cutoff else x)
        
        systolic_BP = df[systolic_cols].mean(axis=1)
        df[systolic_col] = systolic_BP
        high_systolic_BP = systolic_BP.apply(lambda x: 0 if x < SBP_cutoff else x)
        high_systolic_BP = high_systolic_BP.apply(lambda x: 1 if x >= SBP_cutoff else x)
        
        df[htn_exam_col] = pd.concat([high_systolic_BP, high_diastolic_BP], axis=1).sum(axis=1, min_count=1) * valid_BP
        df[htn_exam_col] = df[htn_exam_col].apply(lambda x: 1 if x > 0 else x)
        
        # drop unnecessary cols
        df.drop(NHANES_transformations[htn_exam_col], axis=1, inplace=True)
        
    if 'BPQ_H.xpt' in dataset_path: # blood pressure prescription self-reported        
        df['BPQ040A'].replace(binary_response_dict, inplace=True) # taking HTN prescription
        df['BPQ050A'].replace(binary_response_dict, inplace=True) # taking high BP prescription
        
        df[htn_prescription_col] = df[['BPQ040A', 'BPQ050A']].sum(axis=1, min_count=1)
        df[htn_prescription_col] = df[htn_prescription_col].apply(lambda x: 1 if x > 0 else x)
        
        # drop unnecessary cols
        df.drop(NHANES_transformations[htn_prescription_col], axis=1, inplace=True)
        
    if 'DEMO_H.xpt' in dataset_path: # demographics
        df[race_ethnicity_col] = df['RIDRETH3'].replace({'':float('nan')}) # Race/ethnicity
        df[gender_col] = df['RIAGENDR'].replace({1:0, 2:1}) # Male, Female
        df[age_col] = df['RIDAGEYR'].astype(float) # Age
        df[income_col] = df['INDFMPIR'] # Ratio of family income to poverty
        df[marital_col] = df['DMDMARTL'].replace({77:float('nan'), 99:float('nan'), '':float('nan')}) # marital/partner status
        
        # drop unnecessary cols
        df.drop(['RIDRETH3', 'RIAGENDR', 'RIDAGEYR', 'INDFMPIR', 'DMDMARTL'], axis=1, inplace=True)
    
    if 'DIQ_H.xpt' in dataset_path: # diabetes interview
        # calculate diabetes
        df['DIQ280'].replace({777:float('nan'), 999:float('nan'), '':float('nan')}, inplace=True)
        df['DIQ280'] = df['DIQ280'].apply(lambda x: 0 if x < A1c_cutoff else x)
        df['DIQ280'] = df['DIQ280'].apply(lambda x: 1 if x >= A1c_cutoff else x)
        
        df['DIQ070'].replace(binary_response_dict, inplace=True) # taking diabetic pill 
        df['DIQ050'].replace(binary_response_dict, inplace=True) # insulin
        df['DIQ010'].replace(binary_response_dict, inplace=True) # diabetes dx
        
        df[diabetes_col] = df[['DIQ280','DIQ070','DIQ050','DIQ010']].sum(axis=1, min_count=1)
        df[diabetes_col] = df[diabetes_col].apply(lambda x: 1 if x > 0 else x)
        
        # calculate hypertension
        bp_response_dict = {7777:0, 9999:0, '':0} # refused, don't know, missing
        
        df["DIQ300D"].replace(bp_response_dict, inplace=True) # recent DBP
        df["DIQ300D"] = df["DIQ300D"].apply(lambda x: 0 if x < DBP_cutoff else x)
        df["DIQ300D"] = df["DIQ300D"].apply(lambda x: 1 if x >= DBP_cutoff else x)
        
        df["DIQ300S"].replace(bp_response_dict, inplace=True) # recent SBP
        df["DIQ300S"] = df["DIQ300S"].apply(lambda x: 0 if x < SBP_cutoff else x)
        df["DIQ300S"] = df["DIQ300S"].apply(lambda x: 1 if x >= SBP_cutoff else x)
        
        df["DIQ175H"].replace({17:1, '':float('nan')}, inplace=True) # high bp
        
        df[htn_interview_col] = df[["DIQ300D", "DIQ300S", "DIQ175H"]].sum(axis=1, min_count=1)
        df[htn_interview_col] = df[htn_interview_col].apply(lambda x: 1 if x > 0 else x)
        
        # drop unnecessary cols
        df.drop(NHANES_transformations[htn_interview_col] + 
                NHANES_transformations[diabetes_col], axis=1, inplace=True)
    
    if 'HIQ_H.xpt' in dataset_path: # insurance status
        df[insurance_col] = df['HIQ011'].replace(binary_response_dict)

        # drop unnecessary cols
        df.drop(NHANES_transformations[insurance_col], axis=1, inplace=True)
    
    if 'ALQ_H.xpt' in dataset_path: # alcohol use
        alc_avg_quantity = df['ALQ130'].replace(
            {777:float('nan'), 999:float('nan'), '':float('nan')})  # Avg # alcoholic drinks/day when drinking - past 12 mos
        
        alc_frequency = df['ALQ120Q'].replace(
            {777:float('nan'), 999:float('nan'), '':float('nan')}).round()  # How often drink alcohol over past 12 mos
        
        alc_avg_quantity[alc_frequency == 0] = 0
        
        df[alcohol_col] = alc_avg_quantity * alc_frequency # total # drinks/year
        
        # drop unnecessary cols
        df.drop(NHANES_transformations[alcohol_col], axis=1, inplace=True)
    
    if 'MCQ_H.xpt' in dataset_path: # cardiovascular disease
        df[cvd_col] = df[["MCQ160B", # Ever told had congestive heart failure
                          "MCQ160C", # Ever told you had coronary heart disease
                          "MCQ160E", # Ever told you had heart attack
                          "MCQ160D", # Ever told you had angina/angina pectoris
                          "MCQ160F"] # Ever told you had a stroke
                         ].replace(binary_response_dict).sum(axis=1, min_count=1)
        df[cvd_col] = df[cvd_col].apply(lambda x: 1 if x > 0 else x)
        
        # drop unnecessary cols
        df.drop(NHANES_transformations[cvd_col], axis=1, inplace=True)
        
    if 'DPQ_H.xpt' in dataset_path: # depression screener
        # df[depression_col] = pd.cut(df[PHQ_9_cols].round().sum(axis=1, min_count=1),
        #                             bins=PHQ_9_cuttoffs, 
        #                             include_lowest=True,
        #                             labels=[0,1,2]).astype(float)
        
        df[depression_col] = df[PHQ_9_cols].round().sum(axis=1, min_count=1) # raw PHQ-9 summed score
        
        # drop unnecessary columns
        df.drop(PHQ_9_cols, axis=1, inplace=True)
    
    if 'SLQ_H.xpt' in dataset_path: # sleep disorders
        # calculate sleeping troubles
        df['SLQ060'].replace(binary_response_dict, inplace=True) # sleep disorder diagnosis
        df['SLQ050'].replace(binary_response_dict, inplace=True) # reported trouble sleeping
        df[sleep_troubles_col] = df[['SLQ050', 'SLQ060']].sum(axis=1, min_count=1)
        df[sleep_troubles_col][df[sleep_troubles_col] > 0] = 1
        
        # calculate sleep deprivation
        sleep_hours = df['SLD010H'].replace({99: float('nan')})
        df[sleep_deprivation_col] = pd.cut(sleep_hours,
                                           bins=sleep_deprivation_cutoffs, 
                                           include_lowest=True,
                                           labels=[2,1,0]).astype(float)

        # drop unecessary cols
        df.drop(NHANES_transformations[sleep_troubles_col] +
                NHANES_transformations[sleep_deprivation_col] , axis=1, inplace=True)
    
    if 'SMQRTU_H.xpt' in dataset_path: # recent tobacco use
        df[smoker_recent_col] = df['SMQ681'].replace(
            {1:1, 2:0, 7:float('nan'), 9:float('nan'),'':float('nan')}) # yes, no, refused, don't know, missing
        
        # drop unnecessary cols
        df.drop(NHANES_transformations[smoker_col], axis=1, inplace=True, errors='ignore')
        
    if 'SMQ_H.xpt' in dataset_path: # smoking status
        df[smoker_current_col] = df['SMQ040'].replace( 
            {1:1, 2:1, 3:0, 7:float('nan'), 9:float('nan'),'':float('nan')}) # everyday, some days, not at all, refused, don't know, missing
        
        df[smoker_history_col] = df['SMQ020'].replace(binary_response_dict) # > 100 cigarettes in life
        
        # drop unnecessary cols
        df.drop(NHANES_transformations[smoker_col] + 
                NHANES_transformations[smoker_history_col], axis=1, inplace=True, errors='ignore')

    if 'BMX_H.xpt' in dataset_path: # body measures (BMI)
        df[bmi_col] = df['BMXBMI']
        
        # drop unnecessary cols
        df.drop(NHANES_transformations[bmi_col], axis=1, inplace=True)
        
    if 'RXQ_RX_H.xpt' in dataset_path: # prescription medication use
        drug_id_col = 'RXDDRGID' # generic drug code
        drug_use_col = 'RXDUSE' # taken or used any prescription medicines in the past month
        
        # decode from pandas-inferred type of bytes
        df[drug_id_col] = df[drug_id_col].str.decode('utf-8')
        
        # get drug lookup table
        dl_questionnaire_path = 'RXQ_DRUG.xpt'
        mltc_vars = vars_lookup_df.query(f'{q_name} == "{dl_questionnaire_path.replace(".xpt","")}"')['Variable Name'].tolist()
        drug_lookup_df = pd.read_sas(os.path.join(NHANES_dir, dl_questionnaire_path),
                                        encoding='utf-8',
                                        index=drug_id_col)[mltc_vars]
        
        # get dictionary indicating whether drug is in the relevant drug categories
        mh_drug_dict = check_strings_in_columns(drug_lookup_df, mltc_vars, mh_drug_categories)
        gc_drug_dict = check_strings_in_columns(drug_lookup_df, mltc_vars, gc_drug_categories)
        assert pd.DataFrame.from_dict(mh_drug_dict, orient='index').sum().item() > 0, f'None of {mh_drug_categories} found in the drug lookup df'
        assert pd.DataFrame.from_dict(gc_drug_dict, orient='index').sum().item() > 0, f'None of {gc_drug_categories} found in the drug lookup df'

        # get indicator variable for use of drug categories
        mh_drug_id = df[drug_id_col].replace({'':float('nan'), **mh_drug_dict})
        gc_drug_id = df[drug_id_col].replace({'':float('nan'), **gc_drug_dict})
        drug_use = df[drug_use_col].replace(binary_response_dict)
        
        # create new df
        df = pd.DataFrame()
        df[mh_drug_col] = (mh_drug_id * drug_use).groupby(index).sum() > 0
        df[gc_drug_col] = (gc_drug_id * drug_use).groupby(index).sum() > 0
        df = df.astype(int)
        
    return df


def preprocess_NHANES(exclude: list, q_name='Questionnaire'):   
    # set loop counter
    i = 0 
    
    # read in variable lookup df
    vars_lookup_df = pd.read_csv(os.path.join(data_dir, NHANES_vars_lookup_filename))
    
    # get questionnaire names
    relevant_questionnaires = vars_lookup_df['Data File Name'].apply(lambda x: re.findall('\(([^)]+)', x)[0])
    vars_lookup_df[q_name] = relevant_questionnaires

    # get available datasets
    dataset_filenames = os.listdir(NHANES_dir)
    
    # check datasets cover the relevant questionnaires
    for questionnaire in relevant_questionnaires:
        assert any([questionnaire in x for x in dataset_filenames]), f'The dataset file for {questionnaire} is missing.'
    
    # for all questionnaires
    for dataset_filename in dataset_filenames:
        if dataset_filename in exclude:
            continue
        
        print(f'Preprocessing {dataset_filename}...')

        # get relevant variables
        columns = vars_lookup_df.query(f'{q_name} == "{dataset_filename.replace(".xpt","")}"')['Variable Name'].tolist()
        
        # read in data
        q_path = os.path.join(NHANES_dir, dataset_filename)
        if i == 0:
            df = get_NHANES_questionnaire_df(vars_lookup_df, q_path, columns)
        else:
            temp_df = get_NHANES_questionnaire_df(vars_lookup_df, q_path, columns)
            
            # concatenate it
            df = pd.concat([df, temp_df], axis=1)
        
        i += 1
    
    # combine variables
    htn_cols = [htn_exam_col, htn_interview_col, htn_prescription_col]
    smoker_cols = [smoker_recent_col, smoker_current_col]
    
    for main_col, sub_cols in [(htn_col, htn_cols), 
                               (smoker_col, smoker_cols)]:
        
        if all([col in df.columns for col in sub_cols]):
            df[main_col] = df[sub_cols].sum(axis=1, min_count=1)
            df[main_col] = df[main_col].apply(lambda x: 1 if x > 0 else x)
            
            # drop unnecessary cols
            df.drop(sub_cols, axis=1, inplace=True)
    
    return df

def download_nhanes_xpt(url_list, download_dir='../data/NHANES'):
    """
    Downloads a list of .xpt files from URLs if they don't already exist in the specified directory.

    Args:
        url_list (list): A list of URLs pointing to .xpt files.
        download_dir (str): The directory to download the files to. Defaults to 'NHANES'.
    """

    if not os.path.exists(download_dir):
        os.makedirs(download_dir)

    for url in url_list:
        parsed_url = urlparse(url)
        filename = os.path.basename(parsed_url.path)
        filepath = os.path.join(download_dir, filename)

        if not os.path.exists(filepath):
            try:
                print(f"Downloading {filename}...")
                response = requests.get(url, stream=True)
                response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)

                with open(filepath, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                print(f"Downloaded {filename} successfully.")

            except requests.exceptions.RequestException as e:
                print(f"Error downloading {filename}: {e}")
            except IOError as e:
                print(f"Error writing to file {filename}: {e}")
        else:
            print(f"{filename} already exists. Skipping.")