## NHANES
# set transformation for binary responses
binary_response_dict = {1:1, 2:0, 7:float('nan'), 9:float('nan'), '':float('nan')} # Yes, No, Refused, Don't Know, Missing

# set revelant drugs 
mh_drug_categories = ['ANTIDEPRESSANTS', "ANXIOLYTICS"]
gc_drug_categories = ['GLUCOCORTICOIDS']

# Set PHQ-9 cols
PHQ_9_cols = ["DPQ010","DPQ010","DPQ020","DPQ030","DPQ040","DPQ050","DPQ060","DPQ070","DPQ080","DPQ090"]

# set cutoffs
DBP_cutoff = 80
SBP_cutoff = 130
A1c_cutoff = 6.5
PHQ_9_cuttoffs = [0,4,14,27] # minimal, mild/moderate, moderately severe/severe
sleep_deprivation_cutoffs = [0,5,7,24] # severe moderate, normal
age_cutoff = 35

# set aggregated column names
htn_col = 'HTN'
htn_interview_col = f'{htn_col}_interview'
htn_exam_col = f'{htn_col}_exam'
htn_prescription_col = f'{htn_col}_prescription'
systolic_col = 'SBP'
diastolic_col = 'DBP'
sleep_deprivation_col = 'sleep_deprivation'
light_col = 'ambient_light'

diabetes_col = 'diabetes'
depression_col = 'depression'
cvd_col = 'CVD'
mh_drug_col = '_'.join(mh_drug_categories)
gc_drug_col = gc_drug_categories[0]
sleep_troubles_col = 'sleep_troubles'

physical_activity_col = 'physical_activity'
sedentary_col = 'daily_sedentary'
accelerometer_col = 'accelerometer'

race_ethnicity_col = 'race_ethnicity'
gender_col = 'gender'
age_col = 'age'
smoker_col = 'smoker'
smoker_recent_col = f'{smoker_col}_recent'
smoker_current_col = f'{smoker_col}_current'
smoker_history_col = f'{smoker_col}_hx'
alcohol_col = 'yearly_alcohol'
bmi_col = 'BMI'
income_col = 'poverty_ratio'
marital_col = 'martial_status'
insurance_col = 'health_insurance'
    
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
    cvd_col:["MCQ160B","MCQ160C","MCQ160E","MCQ160D","MCQ160F"],
    sleep_troubles_col:['SLQ060', 'SLQ050'],
        
    physical_activity_col:['PAQ620','PAQ665','PAQ605','PAQ650'],
    sedentary_col:['PAD680'],
    accelerometer_col:['PAXMTSD'],
    
    race_ethnicity_col:['RIDRETH3'],
    gender_col:['RIAGENDR'],
    age_col:['RIDAGEYR'],
    smoker_col:['SMQ040', 'SMQ681'],
    smoker_history_col: ['SMQ020'],
    bmi_col:['BMXBMI'],
    income_col:['INDFMPIR'],
    marital_col:['DMDMARTL'],
    insurance_col:['HIQ011'],
    alcohol_col: ['ALQ130', 'ALQ120Q']
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

# define column data types
categorical_cols = [
       physical_activity_col,
       mh_drug_col, 
       gc_drug_col,
       sleep_troubles_col,
       sleep_deprivation_col,
       diabetes_col,
       smoker_col,
       smoker_history_col,
       race_ethnicity_col,
       gender_col,
       htn_col,
       cvd_col,
       marital_col,
       insurance_col,
]

numerical_cols = [
       depression_col,
       sedentary_col,
       accelerometer_col,
       bmi_col,
       age_col,
       income_col,
       diastolic_col,
       systolic_col
]