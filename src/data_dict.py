## NHANES
# set transformation for binary responses
binary_response_dict = {1:1, 2:0, 7:0, 9:0, '':0} # Yes, No, Refused, Don't Know, Missing

# set revelant mental health drugs 
mh_drug_categories = ['ANTIDEPRESSANTS', "ANXIOLYTICS"]

# Set PHQ-9 cols
PHQ_9_cols = ["DPQ010","DPQ010","DPQ020","DPQ030","DPQ040","DPQ050","DPQ060","DPQ070","DPQ080","DPQ090"]

# set aggregated column names
hypertension_prefix = 'HTN'
htn_interview_col = f'{hypertension_prefix}_interview'
htn_exam_col = f'{hypertension_prefix}_exam'
htn_prescription_col = f'{hypertension_prefix}_prescription'
sleep_deprivation_col = 'sleep_deprivation'
light_col = 'ambient_light'

diabetes_col = 'diabetes'
depression_col = 'depression'
mh_drug_col = '_'.join(mh_drug_categories)
sleep_troubles_col = 'sleep_troubles'

physical_activity_col = 'physical_activity'
sedentary_col = 'daily_sedentary'
accelerometer_col = 'accelerometer'

race_ethnicity_col = 'race_ethnicity'
gender_col = 'gender'
age_col = 'age'
smoker_col = 'smoker'
bmi_col = 'BMI'
income_col = 'poverty_ratio'
    
# dict of column to NHANES transformation variables 
NHANES_transformations = {
    htn_interview_col:['DIQ280','DIQ070','DIQ050','DIQ010'],
    htn_prescription_col:['BPQ040A','BPQ050A'],
    htn_exam_col:['PEASCST1','BPXSY1','BPXSY2','BPXSY3','BPXSY4','BPXDI1','BPXDI2','BPXDI3','BPXDI4'],
    sleep_deprivation_col:['SLD010H'],
    light_col:['PAXPREDM', 'PAXLXMM'],

    diabetes_col:["DIQ300D","DIQ300S","DIQ175H"],
    depression_col:PHQ_9_cols,
    mh_drug_col:['RXDDRGID', 'RXDUSE'],
    sleep_troubles_col:['SLQ060', 'SLQ050'],
        
    physical_activity_col:['PAQ620','PAQ665','PAQ605','PAQ650'],
    sedentary_col:['PAD680'],
    accelerometer_col:['PAXMTSD'],
    
    race_ethnicity_col:['RIDRETH3'],
    gender_col:['RIAGENDR'],
    age_col:['RIDAGEYR'],
    smoker_col:['SMQ040'],
    bmi_col:['BMXBMI'],
    income_col:['INDFMPIR']
}

NHANES_nan_fill = {
    htn_interview_col: 0,
    htn_prescription_col:0,
    htn_exam_col:0,
    sleep_deprivation_col:0,
    light_col:float('nan'),

    diabetes_col:0,
    depression_col:0,
    mh_drug_col:0,
    sleep_troubles_col:0,
        
    physical_activity_col:0,
    sedentary_col:float('nan'),
    accelerometer_col:float('nan'),

    race_ethnicity_col:float('nan'),
    gender_col:float('nan'),
    age_col:float('nan'),
    smoker_col:0,
    bmi_col:float('nan'),
    income_col:float('nan'),
}