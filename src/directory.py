import os

# set data dir
data_dir = '../data'
log_dir = '../logs'

for _dir in [log_dir]:
    if not os.path.isdir(_dir):
        os.makedirs(_dir)

NHANES_dir = os.path.join(data_dir, 'NHANES')
NHANES_vars_lookup_filename = 'NHANES Variables - Sleep Deprivation & HTN - Sheet1.csv'
NHANES_preprocessed_filename = 'NHANES_preprocessed.csv'
