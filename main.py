import numpy as np
np.random.seed(17)
import tensorflow as tf
tf.random.set_seed(17)
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()
import pandas as pd
from spec_augment import SpecAugment
from tensorflow.data import Dataset
from tensorflow import keras
import tensorflow.keras.backend as K
import tensorflow_addons as tfa
import tensorflow_probability as tfp 
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import *  
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import Mean, AUC, BinaryAccuracy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

DF = pd.read_csv('/home/aga/Desktop/final/DF.csv')
VAL = pd.read_csv('/home/aga/Desktop/final/VAL_PAIRS.csv')
TEST = pd.read_csv('/home/aga/Desktop/final/TEST_PAIRS.csv')

phq_dict = {}
for i,x in enumerate(DF.librosa_mel.values):
    phq_dict[x] = DF.T0_total.values[i]
    
for i in ['D','H']:
    VAL[i+'y'] = [phq_dict[x] for x in VAL[i].values]    
    TEST[i+'y'] = [phq_dict[x] for x in TEST[i].values]
    
ws_id = DF.id.value_counts().index[0]
ONE_ID = DF.loc[DF.id==ws_id]
ONE_ID_PAIRS = pd.merge(ONE_ID,ONE_ID, on=['id','site','gender'], how='inner')
ONE_ID_PAIRS = ONE_ID_PAIRS.rename(columns={'T0_total_x':'Dy','T0_total_y':'Hy','librosa_mel_x':'D','librosa_mel_y':'H'})
ONE_ID_PAIRS['phq_diff'] = ONE_ID_PAIRS.Dy - ONE_ID_PAIRS.Hy
ONE_ID_PAIRS = ONE_ID_PAIRS.loc[ONE_ID_PAIRS.phq_diff>2]

df = DF.loc[DF.id!=ws_id]
VAL = VAL.loc[VAL.id!=ws_id]
TEST = TEST.loc[TEST.id!=ws_id]

val_files = list(VAL.D.unique())+list(VAL.H.unique())
test_files = list(TEST.D.unique())+list(TEST.H.unique())
random_files = np.random.choice(list(df.librosa_mel.values),1000)
holdout_files = val_files+test_files+list(random_files)

train_df = df.loc[~df.librosa_mel.isin(holdout_files)].copy()
random_holdouts = df.loc[df.librosa_mel.isin(random_files)]

train_ids = list(train_df.id.unique())
val_ids = list(VAL.id.unique())

train_df['familiar'] = train_df.id.isin(val_ids)
VAL['familiar'] = VAL.id.isin(train_ids)

cols = ['librosa_mel','T0_total','site','gender','id','familiar','task']

TRAIN = pd.merge(train_df[cols],train_df[cols], on=['id','gender','site','familiar'], how='inner')
TRAIN = TRAIN.rename(columns={'T0_total_x':'Dy','T0_total_y':'Hy','librosa_mel_x':'D','librosa_mel_y':'H'})

TRAIN['phq_diff'] = TRAIN.Dy - TRAIN.Hy
VAL['phq_diff'] = VAL.Dy - VAL.Hy
TEST['phq_diff'] = TEST.Dy - TEST.Hy

TRAIN = TRAIN.loc[TRAIN.phq_diff>=5]
VAL = VAL.loc[VAL.phq_diff>=3]

PRODROME_TEST = pd.concat([VAL.loc[(VAL.Dy<4)&(VAL.Hy<4)],TEST.loc[(TEST.Dy<4)&(TEST.Hy<4)]])
FAMILIAR_TEST = TEST.loc[TEST.id.isin(TRAIN.id.unique())]

cf_train = make_ds(TRAIN, ws=True)
naive_train = make_ds(train_df, ws=True)
dual_val = make_dual_val(VAL, random_holdouts)
test = make_dual_val(TEST, random_holdouts)

prodrome_test = make_dual_val(PRODROME_TEST, random_holdouts)
familiar_test = make_dual_val(FAMILIAR_TEST, random_holdouts)

cf_esn = make_cf(base_ESN())
cf_esn, cf_history, cf_test = run_trial(cf_esn, train=cf_train, val=dual_val, test=[test,prodrome_test,familiar_test])

naive_esn = make_naive(base_ESN())
naive_esn, naive_history, naive_test = run_trial(naive_esn, train=naive_train, val=dual_val, test=[test,prodrome_test,familiar_test])
