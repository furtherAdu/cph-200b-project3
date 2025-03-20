import os
import re
import requests
from urllib.parse import urlparse
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from scipy.stats import spearmanr, norm

from src.directory import data_dir, NHANES_dir, NHANES_vars_lookup_filename
from src.data_dict import NHANES_transformations, binary_response_dict, NHANES_nan_fill
from src.data_dict import htn_exam_col, htn_prescription_col, htn_interview_col, physical_activity_col, accelerometer_col,\
    race_ethnicity_col, gender_col, age_col,smoker_col, income_col, depression_col, sleep_deprivation_col, sleep_troubles_col,\
    PHQ_9_cols, bmi_col, mh_drug_categories, mh_drug_col, diabetes_col, sedentary_col


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

def preprocess_ist(data, feature_cols, untransformed_cols, heparin_cols, combined_hep_col='DH14'):
    # handle 'U' for RCONSC
    data['RCONSC'] = data['RCONSC'].replace({'U':'unc'})

    # map strings to numeric
    data = data.replace({'Y':1, 'y':1, 'N':0, 'n':0, 'C':float('nan'), 'U': float('nan')})

    # combine heparin columns
    data[combined_hep_col] = data[heparin_cols].sum(axis=1) > 0
    data.drop(heparin_cols, axis=1, inplace=True)

    # get columns by data type
    untransformed_cols.append(combined_hep_col)
    categorical_cols = list(set(data.columns[data.apply(lambda x: x.nunique() < 10)]) - set(untransformed_cols))
    numerical_cols = list(set(data.columns) - set(categorical_cols + untransformed_cols))

    # one-hot encode categorical data
    data = one_hot(data, categorical_cols)

    # standardize numerical data
    scaler = StandardScaler()
    data[numerical_cols] = scaler.fit_transform(data[numerical_cols])

    # get covariates
    covariates = [x for x in data.columns for y in feature_cols if x.startswith(y)]

    return data, covariates

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

def get_NHANES_questionnaire_df(vars_lookup_df, dataset_path, columns, index='SEQN', q_name='Questionnaire'):
    # read in dataframe
    encoding = "utf-8" if 'RXQ_RX_H.xpt' not in dataset_path else None
    df = pd.read_sas(dataset_path, index=index, encoding=encoding)[columns]
    
    if 'PAQ_H.xpt' in dataset_path: # physical activity self-reported
        vigorous_moderate_activity_cols = NHANES_transformations[physical_activity_col]
        
        # get indicator variable
        df[vigorous_moderate_activity_cols] = df[vigorous_moderate_activity_cols].replace(binary_response_dict).fillna(0)
        df[physical_activity_col] = (df[vigorous_moderate_activity_cols].sum(axis=1) > 0).astype(int)
        
        df[sedentary_col] = df['PAD680'].replace({7777:float('nan'), 
                                                  9999:float('nan')}) / (24 * 60)
        
        # drop unnecessary columns
        df.drop(vigorous_moderate_activity_cols + 
                NHANES_transformations[sedentary_col], axis=1, inplace=True)
    
    if 'PAXDAY_H.xpt' in dataset_path: # day sum of accelerometer triaxial values
        df = df.groupby(index).mean().rename(
            columns={NHANES_transformations[accelerometer_col][0]:accelerometer_col})
    
    if 'PAXMIN_H.xpt' in dataset_path: # ambient light
        total_minutes = df.groupby(index).count()
        sleep_wear = df.query('PAXPREDM==2')
        lux_value = sleep_wear['PAXLXMM'].groupby(index).sum(axis=1)
        # quality_flag = sleep_wear['PAXFLGSM']
        
        df = lux_value / total_minutes
        
    
    if 'BPX_H.xpt' in dataset_path: # blood pressure examination
        valid_BP = df['PEASCST1'].replace({2:1, 3:0, '':0}) # partial, not done, missing
        systolic_cols = ['BPXSY1','BPXSY2','BPXSY3','BPXSY4']
        diastolic_cols = ['BPXDI1','BPXDI2','BPXDI3','BPXDI4']
        
        systolic_BP = df[systolic_cols].mean(axis=1).fillna(0)
        diastolic_BP = df[diastolic_cols].mean(axis=1).fillna(0)
        
        df[htn_exam_col] = (((systolic_BP > 130) + (diastolic_BP > 80)) > 0) * valid_BP
        
        # drop unnecessary cols
        df.drop(NHANES_transformations[htn_exam_col], axis=1, inplace=True)
        
    if 'BPQ_H.xpt' in dataset_path: # blood pressure prescription self-reported        
        taking_HTN_prescription = df['BPQ040A'].replace(binary_response_dict) > 0
        taking_high_BP_prescription = df['BPQ050A'].replace(binary_response_dict) > 0
        
        df[htn_prescription_col] = ((taking_HTN_prescription + taking_high_BP_prescription) > 0).astype(int).fillna(0) 
        
        # drop unnecessary cols
        df.drop(NHANES_transformations[htn_prescription_col], axis=1, inplace=True)
        
    if 'DEMO_H.xpt' in dataset_path: # demographics
        df[race_ethnicity_col] = df['RIDRETH3'].replace({'':0}) # Race/ethnicity
        df[gender_col] = df['RIAGENDR'].replace({1:0, 2:1}) # Male, Female
        df[age_col] = df['RIDAGEYR'].astype(float) # Age
        df[income_col] = df['INDFMPIR'] # Ratio of family income to poverty
        
        # drop unnecessary cols
        df.drop(['RIDRETH3', 'RIAGENDR', 'RIDAGEYR', 'INDFMPIR'], axis=1, inplace=True)
    
    if 'DIQ_H.xpt' in dataset_path: # diabetes interview
        # calculate diabetes
        A1c_cutoff = 6.5
        above_threshold_A1c = (df['DIQ280'].replace({777:0, 999:0, '':0})  # refused, don't know, missing
                                >= A1c_cutoff).astype(int) 
        diabetic_pill = df['DIQ070'].replace(binary_response_dict) 
        insulin = df['DIQ050'].replace(binary_response_dict)
        diabetes_diagnosis = df['DIQ010'].replace(binary_response_dict)
        
        df[diabetes_col] = ((above_threshold_A1c + diabetic_pill + insulin + diabetes_diagnosis) > 0).astype(int)
        
        # calculate hypertension
        bp_response_dict = {7777:0, 9999:0, '':0} # refused, don't know, missing
        recent_DBP = df["DIQ300D"].replace(bp_response_dict)
        recent_SBP = df["DIQ300S"].replace(bp_response_dict)
        high_BP = df["DIQ175H"].replace({17:1, '':0}).fillna(0) # high bp, missing
        df[htn_interview_col] = (((recent_SBP > 130) + (recent_DBP > 80) + high_BP) > 0).astype(int)
        
        # drop unnecessary cols
        df.drop(NHANES_transformations[htn_interview_col] + 
                NHANES_transformations[diabetes_col], axis=1, inplace=True)
        
    if 'DPQ_H.xpt' in dataset_path: # depression screener
        depression_cutoffs = [0,4,14,27] # minimal, mild/moderate, moderately severe/severe
        df[depression_col] = pd.cut(df[PHQ_9_cols].sum(axis=1),
                                    bins=depression_cutoffs, 
                                    labels=[0,1,2]).fillna(0).astype(int)
        
        # drop unnecessary columns
        df.drop(PHQ_9_cols, axis=1, inplace=True)
    
    if 'SLQ_H.xpt' in dataset_path: # sleep disorders
        # calculate sleeping troubles
        sdisorder_diagnosis = df['SLQ060'].replace(binary_response_dict)
        reported_trouble_sleeping = df['SLQ050'].replace(binary_response_dict)
        df[sleep_troubles_col] = ((sdisorder_diagnosis + reported_trouble_sleeping) > 0).astype(int)
        
        # calculate sleep deprivation
        sleep_deprivation_cutoffs = [0,5,7,24] # severe moderate, normal
        sleep_hours = df['SLD010H'].replace({99: max(sleep_deprivation_cutoffs),
                                             float('nan'): max(sleep_deprivation_cutoffs)})
        df[sleep_deprivation_col] = pd.cut(sleep_hours,
                                           bins=sleep_deprivation_cutoffs, 
                                           labels=[2,1,0]).astype(int)

        # drop unecessary cols
        df.drop(NHANES_transformations[sleep_troubles_col] +
                NHANES_transformations[sleep_deprivation_col] , axis=1, inplace=True)
        
    if 'SMQ_H.xpt' in dataset_path: # cigarette use
        df[smoker_col] = df['SMQ040'].replace(
            {1:2, 2:1, 3:0, 7:0, 9:0,'':0}) # everyday, some days, not at all, refused, don't know, missing
        
        # drop unnecessary cols
        df.drop(NHANES_transformations[smoker_col], axis=1, inplace=True)

    if 'BMX_H.xpt' in dataset_path: # body measures (BMI)
        df[bmi_col] = df['BMXBMI']
        
        # drop unnecessary cols
        df.drop(NHANES_transformations[bmi_col], axis=1, inplace=True)
        
    if 'RXQ_RX_H.xpt' in dataset_path: # prescription medication use
        drug_id_col = 'RXDDRGID'
        drug_use_col = 'RXDUSE'
        
        # decode from pandas-inferred type of bytes
        df[drug_id_col] = df[drug_id_col].str.decode('utf-8')
        
        # get drug lookup table
        dl_questionnaire_path = 'RXQ_DRUG.xpt'
        mltc_vars = vars_lookup_df.query(f'{q_name} == "{dl_questionnaire_path.replace(".xpt","")}"')['Variable Name'].tolist()
        drug_lookup_df = pd.read_sas(os.path.join(NHANES_dir, dl_questionnaire_path),
                                        encoding='utf-8',
                                        index=drug_id_col)[mltc_vars]
        
        # get dictionary indicating whether drug is in the relevant drug categories
        relevant_drug_dict = check_strings_in_columns(drug_lookup_df, mltc_vars, mh_drug_categories)
        assert pd.DataFrame.from_dict(relevant_drug_dict, orient='index').sum().item() > 0, f'None of {mh_drug_categories} found in the drug lookup df'
        
        # get indicator variable for use of drug categories
        df[drug_id_col] = df[drug_id_col].replace({'':'0', **relevant_drug_dict}).astype(int)
        df[drug_use_col] = df[drug_use_col].replace(binary_response_dict)
        
        # create new df
        df = ((df[drug_id_col] * df[drug_use_col]).groupby(index).sum() > 0).astype(int).to_frame(name=mh_drug_col)
        
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
    
    # fill nan
    for col, fill_value in NHANES_nan_fill.items():
        if col in df.columns:
            df[col] = df[col].fillna(fill_value)
    
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