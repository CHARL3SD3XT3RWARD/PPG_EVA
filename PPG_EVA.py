# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 14:37:23 2025

@author: alko18
"""

#import pyedflib as plib
#import os
import ast
import warnings
#import PPG_EVA_GUI as eva

import numpy as np
#import time
import matplotlib.pyplot as plt
import pandas as pd
import scipy as sc
import scipy.signal as scsignal
from scipy.stats import entropy, skew, kurtosis
from scipy.interpolate import make_interp_spline as C2_Spline #C2 Continuity
from scipy.interpolate import PchipInterpolator as pchip
from scipy.ndimage import gaussian_filter
from sklearn.metrics import precision_recall_curve, average_precision_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import roc_curve, auc
from collections import defaultdict
import os
import pickle

import matplotlib as mpl
#%% config default settings

ordner_pfad =r'A:\example\directory'# basedirectory to the signals
decision_pfad = None #Deprecated
training_signal = None
annotation = None
training_values_path = r'A:\my\directory\all_data_tuples.xlsx' # filedirectory to the training data
fs_A = 128
fs_B = 32
signal_length = 15
order = 2
lowcut = .5
highcut = 8
chunk_length = 1 #min
low_BPM = 48
high_BPM = 120
TN_list=None
      
date_format = "%d.%m.%Y %H:%M:%S,%f"

mpl.rcParams.update({
    "figure.figsize": (6, 4),
    "axes.titlesize": 20,
    "axes.labelsize": 15,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 15,
    "lines.linewidth": 2,
    "axes.grid": True,
    "grid.alpha": 0.7,
    "font.family": "serif",  # Oder "sans-serif"
    "savefig.dpi": 300
})

#%% classes
def lin_reg(signal_chunks, cross_pos):  
    '''
    A funktion wicht performs a lin. regression for every given signalchunk.
    
    Parameters
    ----------
    signal_chunks : array
        The sequenced signal.
    cross_pos : 2D-list
        The Position of each zero crossing for every signal_chunks
        
    Returns
    -------
    slope : list
        The slope of every lin. regression.
    intersect : list
        The y-intercept for every lin. regression.
        
    '''
    slope=[]
    intersect=[]
    for idx in range(len(signal_chunks)):
        if len(cross_pos[idx])!=0: # so no errors occur if the chunk has no zero crossings
            x=np.arange(0, len(cross_pos[idx]), 1)       
            fit=sc.stats.linregress(x, cross_pos[idx])
        
            slope.append(fit[0])
            intersect.append(fit[1])
        
        else:
            slope.append(np.nan)
            intersect.append(np.nan)

    return slope, intersect

def variance(slopes, intersect, cross_pos):
    '''
    A function wich calculates the variance of the data relative to the lin. regression.
    
    Parameters
    ----------
    slopes : list
        The slopes for every lin. regression.
    intersect : list
        The y-intersect for ever lin. regression.
    cross_pos : list
        The position of every zero crossing as an array for every signalchunk.
    Returns
    -------
    var : list
        The normalised variance of the data relative to the lin. regression
    '''
    var=[]
    for i in range(len(slopes)):
        if not np.isnan(slopes[i]): #if no zerocrossing occured, the value is NaN
        
            x=np.arange(0, len(cross_pos[i]), 1)
            mu_i=[slopes[i] * x + intersect[i]][0]
            
            var.append(np.mean((cross_pos[i]-mu_i)**2))
        
        else:
            var.append(np.nan)
        
    return var

def import_training_data(training_values_path):# check in pipeline    
    '''
    Imports a excel-sheet with all data of Somno.
    Futhermore, a rondomization and subdivison into test- and trainingsets is performed.   
    
    Parameters
    ----------
    training_values_path : string
        The specific path of the excel-sheet.
        
    Example: A:\my\training\data\training_data.xlsx
    
    Returns
    -------
    validation_sets: dict
        Five subsets with a raugh equal amount of good and bad data.    
    
    training_sets: dict
        Five subsets wich contain four validation_sets. In every training_set one validation_set is missing.
        
    test_set: array
        One set wich contains 20% of the whole dataset with respect to the subdivision in good and bad data.        
        
    
    '''
    
    
    #importin data
    df_data = pd.read_excel(training_values_path)
    df_data.set_index('Unnamed: 0', drop=True, inplace=True)
    
    #separating data
    df_good_data = df_data[df_data['Annotation'] == 1].reset_index(drop=True)
    df_bad_data = df_data[df_data['Annotation'] == 0].reset_index(drop=True)
    
    good_data = df_good_data.to_numpy()
    bad_data = df_bad_data.to_numpy()
    
    #random stuff
    good_idx = np.arange(0,len(good_data), 1)
    bad_idx = np.arange(0,len(bad_data), 1)
    
    np.random.shuffle(good_idx)
    np.random.shuffle(bad_idx)
    
    #taking 20%  for the testset e.g. unseen data for performancetesting
    len_test_idx_g = np.ceil(len(good_data)*0.2).astype(int)
    len_test_idx_b = np.ceil(len(bad_data)*0.2).astype(int)
    
    test_idx_g = good_idx[:len_test_idx_g]
    test_idx_b = bad_idx[:len_test_idx_b]
    
    #initialising testset
    test_set = np.concatenate((good_data[test_idx_g], bad_data[test_idx_b]), axis=0)
    
    #splitting the ramaining data into five subsets
    split_remain_g = np.array_split(good_idx[len_test_idx_g:], 5)
    split_remain_b = np.array_split(bad_idx[len_test_idx_b:], 5)
    
    splits = defaultdict(dict)
    
    for i, good_split in enumerate(split_remain_g):
        bad_split = split_remain_b[i]
        splits[f'subset {i}'] = {'good': good_split, 'bad': bad_split} 
    
    #filling sets
    training_sets = defaultdict(dict)
    validation_sets = defaultdict(dict)
    
    
    for subset in splits:
        val_idx_g = splits[subset]['good']
        val_idx_b = splits[subset]['bad']
        validation_sets[subset] = np.concatenate((good_data[val_idx_g], bad_data[val_idx_b]), axis=0)
        tr_sets_idx_g = np.array([])
        tr_sets_idx_b = np.array([])
        outer_keys = [k for k in splits if k != subset]
        for key in outer_keys:
            tr_sets_idx_g = np.append(tr_sets_idx_g, splits[key]['good']).astype(int)
            tr_sets_idx_b = np.append(tr_sets_idx_b, splits[key]['bad']).astype(int)
        
        training_sets[subset] = {'good': good_data[tr_sets_idx_g], 'bad': bad_data[tr_sets_idx_b]}
   
    return training_sets, validation_sets, test_set

def import_decisionpoints():
    '''
    !!!Deprecated!!!

    Was created to import decision thresholds. 
    Since the classifier is exported with its best threshold this function is obsolete.

    Returns
    -------
    None.

    '''
    
    global somno_decision, somno_decision_snr, somno_decision_entropy, somno_decision_zcr, corsano_decision, corsano_decision_snr, corsano_decision_entropy, corsano_decision_zcr
    
    df_decision_points=pd.read_excel(rf'{decision_pfad}')  
    
    somno_decision=df_decision_points['Somno'].apply(ast.literal_eval)
    somno_decision_snr=somno_decision[0][1] #ist eine liste mit [index, threshold] -> idx nur der vollständigkeit halber
    somno_decision_entropy=somno_decision[1][1]
    somno_decision_zcr=somno_decision[2][1]
    
    corsano_decision=df_decision_points['Corsano'].apply(ast.literal_eval)
    corsano_decision_snr=corsano_decision[0][1]
    corsano_decision_entropy=corsano_decision[1][1]
    corsano_decision_zcr=corsano_decision[2][1]    

def read_signal(mod_path, signal_key, time_key, sep=',', skiprows=0, date_format= None, header='infer'): 
    '''
    

    Parameters
    ----------
    mod_path : string
        The modified filepath to the signalfile wit its name as last part.
        Syntax: path + 'filename'
    signal_key : string
        The keyword/number for Pandas.Dataframe signal-column.
            Somno:   signal_key= 2
            corsano: signal_key='value'
    time_key : string
        Das Schlüsselwort für den Pandas.DataFrame um auf die Zeitstempel zuzugreifen.
        The keyword/number for Pandas.Dataframe timestamp-column
        no:   time_key=0
            corsano: time_key='date'
    sep : string, optional
        The seperator used to seperate the columns. Only necessary for Somno. Corsanofiles use the default.    
        The default is ','.
        sep_somno = ';'
    skiprows : integer, optional
        !!!Deprecated!!!
        The number of rows that should be skipped. Only necessary for Somno (skip 6 rows).
        The default is 0.
    date_format : string, optional
        Date-format for the timestamps.
        Only necessary for Somno ("%d.%m.%Y %H:%M:%S,%f")
        The default is None.
        
    Returns
    -------
    signals : array
        The signal as a timeseries.
    timestamps : Series
        the timestamps as pandas.Series.

    '''
    df=pd.read_csv(mod_path, sep=sep, skiprows=skiprows, header=header)
    df=df.reset_index()
    
    signals =df[signal_key].to_numpy()
    timestamps= pd.to_datetime(df[time_key], format=date_format).dt.tz_localize(None)
    
    return signals, timestamps
   
def decide(values, criterion, zcr=False):
    '''
    !!!Deprecated!!!
    Decides weather a value is good or bad.

    Parameters
    ----------
    values : array
        The values of every signalchunk.
    criterion : float
        The decision criterion.
    zcr : bool, optional
        Since the ZCR should be minimimal for a good value, the decision is inversed. The default is False.

    Returns
    -------
    decision : list
        A list of bools. A one (1) for good and a zero (0) for bad.

    '''
    decision=[]
    for i in range(len(values)):
        if zcr:
            if values[i] > criterion:
                decision.append(0)
            else:
                decision.append(1)
        
        else:
            if values[i] < criterion:
                decision.append(0)
            else:
                decision.append(1)
    return decision

def NOR(a, b):
    '''
    !!! Deprecated!!!
    A simple NOR gate.

    Parameters
    ----------
    a : bool
    b : bool
        

    Returns
    -------
    bool


    '''
    if(a == 0) and (b == 0):
        return True
    elif(a == 0) and (b == 1):
        return False
    elif(a == 1) and (b == 0):
        return False
    elif(a == 1) and (b == 1):
        return False
    
def XOR(a, b):
    '''
    A simple XOR gate.

    Parameters
    ----------
    a : bool
    b : bool
        

    Returns
    -------
    bool

    '''
    if a != b:
        return True
    else:
        return False

def ROC(values, refdata, thresholds = None):
    '''
    !!!Deprecated!!!
    An attempt to build a ROC-Funktion. Does not work well.

    Parameters
    ----------
    values : array
        The calculated SQIs.
    refdata : array
        The annotationarray.
            1 = True  = good
            0 = False = bad

    Returns
    -------
    TPR : list
        The True Positive Rate (Sensitivity)
    PPV: list
        The precision.    
    FPR : list
        The False Positive Rate (Specificity)
    thresh: list
        The used thresholds.

    '''
    #check if ground truth contains two classes. If not, ROC doesnt work.
    if len(set(refdata)) < 2:
        return [], [], [], []
    
    #check if values are from integer type. I forgot why i did that
    if type(values) == int:
        return [], [], [], []
        
    reference = list(map(bool, refdata))
    
    # if no threshold is given e.g. while training the subsets, generate thresholds
    # there was an attempt at finding a threshold with a list previously found thresholds, therefore this is not necessaray.
    if thresholds is None:
        thresholds = np.linspace(min(values), max(values) ,1000)
        
    Sensitivity = []
    Specificity = []
    PPV = []
    
    for thresh in thresholds:
        
        #compare values with threshold
        compare_thresh = (values > thresh)
            
        reference = np.asarray(reference).astype(bool)
        compare_thresh = np.asarray(compare_thresh).astype(bool)
       
        #counting
        TP = np.sum((reference == True) & (compare_thresh == True))
        FP = np.sum((reference == False) & (compare_thresh == True))
        FN= np.sum((reference == True) & (compare_thresh == False))
        TN = np.sum((reference == False) & (compare_thresh == False))
        
        #calculating parameters        
        sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
        ppv = TP/(TP+FP) if (TP+FP) > 0 else 0
        
        Sensitivity.append(sensitivity)
        Specificity.append(specificity)
        PPV.append(ppv)        

    return Sensitivity, PPV, Specificity, thresholds                          

def ROC_mid(values):
    '''
    !!!Deprecated!!!
    Calculates a mean ROC for every SQI.
    
    Parameters
    ----------
    values : list of lists
        A list of ROC-data for a specific SQI
    Returns
    -------
    list
        A list of the mean Sensitivity (Index 0), mean specificity (index 1)
 
    '''

    ROC_mid1=[]
    ROC_mid2=[]
        
    #ROC_mid=[np.mean(values[:][0], 1), np.mean(values[:][1], 1)]    
    for i in range(len(values[0][0])):
        temp1= (values[0][0][i] + values[1][0][i] + values[2][0][i])/3
        temp2= (values[0][1][i] + values[1][1][i] + values[2][1][i])/3
        ROC_mid1.append(temp1)
        ROC_mid2.append(temp2)
        
    return [ROC_mid1, ROC_mid2, values[0][2]]    
        
def minimize(values):
    '''
    !!!Deprecated!!!
    Find the value with the minimal distance to the koordinates [0,1]
   
    Parameters
    ----------
    values: list
        a list of lists with sensitivity at index 0, specificity at index 1 and coresponding thresholds at index 2.
    
    Returns
    -------
    TYPE: Int
        The index with the minimal distance to [0,1] and its corresponding threshold
    '''
    
    sensitivity=values[0]
    specificity=values[1]
    thresholds=values[2]
    dist = []
    origin = np.array([0,1])
    for i in range(len(sensitivity)):
        endpoint= np.array([1-specificity[i], sensitivity[i]])
        temp = np.linalg.norm(endpoint-origin)
        dist.append(temp)
        
    return [int(np.argmin(dist)), float(thresholds[np.argmin(dist)])]

def preprocessing(training_signal_path, TN, chunk_length = 1, training = True, plot=True):
    '''
    A preprocessing returns the SQI as objects for each device.

    Parameters
    ----------
    training_signal_path : string
        The baspath to signaldirectory.
    TN : string
        The TN-ID to find the signals within the training_signal_path.
    chunk_length : float, optional
        The chunklength as a Faktor of one minute. The default is 1.
        Example:
            chunk_length = 1/6 = 60s/6 = 10s
            chunk_length = 1.5 = 1.5*60 = 90s -> 1.5 minutes
    training : bool, optional
        Weather PPG-Eva is in training mode or not. If False, corsano will only used for synchronicing.
        Was used for the aqusition of the training_data. The default is True.
    plot : bool, optional
        If True, the whole signal will be plotted. The default is True.

    '''

    #clac signallength. signal_length is in config
    signal_length_A = signal_length*60*fs_A
    chunk_size=int(chunk_length*60*fs_A)     

    #importing corsanesignal
    path=os.path.join(training_signal_path, TN)# rf"{training_signal}\{TN_list[i]}"
    in_path = os.listdir(path)
    corsano_file=[f for f in in_path if f.endswith('.csv')][0]
    corsano_path = os.path.join(path, corsano_file)
    corsano_signals, corsano_timestamps=read_signal(corsano_path, signal_key= 'value', time_key='date')
    # corsano_signals, corsano_timestamps=read_signal(corsano_path, signal_key= 'Signal', time_key='Timestamp')
        
    #importing somnosignal
    somno_file=[f for f in in_path if f.endswith('.txt')][0]
    somno_path = os.path.join(path, somno_file)
        #somno_signals, somno_timestamps=read_signal(somno_path, signal_key='Data:', time_key='index', sep=';', skiprows=6, date_format=date_format)
    # somno_signals, somno_timestamps=read_signal(somno_path, signal_key='Signal', time_key='Timestamp', sep=';')
    somno_signals, somno_timestamps=read_signal(somno_path, signal_key=2, time_key=0, sep=';', header=None)
       
    #sync/cut            
    joined_signals=JoinSignals(somno_signals, corsano_signals, somno_timestamps, corsano_timestamps)
    joined_signals.syncronizing()
        #imput=fs
    signal_length_B = signal_length*60*joined_signals.sync[2]
    joined_signals.cut(signal_length_A=signal_length_A, signal_length_B=signal_length_B)
     
    #processing somno e.g. filter, sequencing and calc SQIs
    global somno_processing #debugg
    somno_processing=Processing(signals=joined_signals.signal_A, timestamps=joined_signals.timestamps_A)  
        
    somno_processing.pleth_filter(fs=fs_A, lowcut=lowcut, highcut=highcut, order=order,)   
        
    #slicing  
    somno_processing.slice_list(chunk_size=chunk_size)                  
    #calc SQI     
    global somno_SQI
    somno_SQI=SQI(somno_processing.signal_chunks)
            
    somno_SQI.skewness()
    somno_SQI.kurt()
    somno_SQI.calc_SNR()
    somno_SQI.ZCR()
    somno_SQI.shanon_entropy()
        
    if not training: #if training == False

        #same processing as somno but with resampling included
        global corsano_SQI 
        corsano_processing = Processing(signals=joined_signals.signal_B, timestamps=joined_signals.timestamps_B)
        corsano_processing.resampling(sampling_rate=joined_signals.sync[2])
        corsano_processing.pleth_filter(fs=fs_A, lowcut=lowcut, highcut=highcut, order=order, resampled=True)# highcut 8, order 2
        corsano_processing.slice_list(chunk_size=chunk_size)

        corsano_SQI=SQI(corsano_processing.signal_chunks)
        
        corsano_SQI=SQI(corsano_processing.signal_chunks)
        corsano_SQI.skewness()
        corsano_SQI.kurt()
        corsano_SQI.calc_SNR()
        corsano_SQI.ZCR()
        corsano_SQI.shanon_entropy()
            
        if plot:
            plt.figure()
            fig, (ax1, ax2)= plt.subplots(nrows=2, sharex=True, figsize=(20,6))
            
            ax1.plot(joined_signals.signal_A)
            ax2.plot(joined_signals.signal_B)
            # chunk=0
            # chunk_names = []
            # chunk_name_pos = []
            # for i in range(len(somno_processing.signal_chunks)):
            #     ax1.vlines(chunk, np.min(joined_signals.signal_A), np.max(joined_signals.signal_A), color='k', linestyle='--')
            #     ax2.vlines(chunk, np.min(joined_signals.signal_B), np.max(joined_signals.signal_B), color='k', linestyle='--')
            #     chunk_names.append(f'Chunk {i}')
            #     chunk_name_pos.append(chunk+((1/2)*chunk_size))
            #     chunk += chunk_size
                
            # ax1.xticks(chunk_name_pos, chunk_names)
            plt.title(TN)
            ax1.grid()
            ax2.grid()
            plt.show()
    


    #     corsano_SQI.decision()
        #     somno_SQI.decision()
            
        return 0

class Processing:
    '''
    The class for processing the Signals. Included are:
        __init__
        pleth_filter
        struckt (DEPRECATED)
        resampling
        slice_list
    
    '''
    def __init__(self, signals, timestamps):
        '''
        

        Parameters
        ----------
        signals : array
            The signal.
        timestamps : Series
            The coresponding timestamps as Pandas.Series Objekt.

        Returns
        -------
        None.

        '''
        self.signals = signals
        self.timestamps = timestamps
        
    def pleth_filter(self, fs, lowcut, highcut, order, resampled=False):
        '''
        Butterwoth IIR filter second order.

        Parameters
        ----------
        fs : Int
            Sampling rate in Hertz. The default is 128 Hz.
        lowcut : float, optional
            The default is 0.5.
        highcut : float, optional
            The default is 8.
        order : Int, optional
            The default is a Bandpass of 2nd order.
        resampled : Bool, optional
            Wether the Signal is resampled. the resampled signal is a different attribute of this object.
            The default is False.

        Returns
        -------
        None.

        '''
        
        sos = scsignal.butter(order, [lowcut, highcut], fs=fs, btype='band', output='sos')
        if resampled:
            self.filtered_signal = scsignal.sosfiltfilt(sos, self.resampled_signal)*(-1)#inverting
            #entfernen des Gleichspannungsanteil
            self.filtered_signal=self.filtered_signal-np.mean(self.filtered_signal)     
        
        else:
            self.filtered_signal = scsignal.sosfiltfilt(sos, self.signals)
            #entfernen des Gleichspannungsanteil
            self.filtered_signal=self.filtered_signal-np.mean(self.filtered_signal)     
    
    def struckt(self, Samplerate=1/fs_B):
        '''
        !!!Deprecated!!!
        Creates new timestamps for a resampled signal.
        Deprecated since new timestamps are created within the resampling-method.

        Parameters
        ----------
        Samplerate : float, optional
            The NEW samplerate.
            The default is 1/128.

        Returns
        -------
        None.

        '''
        
        signal_length=len(self.resampled_signal)
        SampleInScnds=pd.Timedelta(seconds=Samplerate)
        timestamps=[self.timestamps[0], ]
        for i in range(1,signal_length):
            temp=timestamps[i-1]+SampleInScnds
            timestamps.append(temp)
        
        self.new_timestamps = timestamps 
    
    def resampling(self,  sampling_rate, target_rate=fs_A, interpolate_pchip=False):
        '''

        Parameters
        ----------
        sampling_rate : Int, optional
            The original smaplingrate of the signal. The default is 32.
        target_rate: Int
            The target samplinrate of the signal, The dafault is 128.
        interpolate_pchip: bool
            if True, the resampling is performed by the pchip-method. 
            The default is False and hence C2-Splineinterpolation is used.

        '''
        
        # number of samples in original signal
        n_orig = len(self.signals)
        
        # target number of samples
        n_target = int(n_orig * (target_rate / sampling_rate))
        
        # create old and new timeaxis evenly
        t_old = np.linspace(0, 1, n_orig, endpoint=False)
        t_new = np.linspace(0, 1, n_target, endpoint=False)
        
        if interpolate_pchip:
            splineC1=pchip(t_old, self.signals)
            signal_newC1=splineC1(t_new)
            self.resampled_signal=signal_newC1(t_new)
            self.new_timestamps=pd.to_datetime(t_new, unit='s')
    
        else:
            spline=C2_Spline(t_old, self.signals, k=3)
            self.resampled_signal=spline(t_new)
            self.new_timestamps=pd.to_datetime(t_new, unit='s')

        
    def slice_list(self, chunk_size):
        '''
        Sequencing of the signal into desired length.

        Parameters
        ----------
        chunk_size : Int, optional
            The desired length of sequence.

        Returns
        -------
        None.

        '''
        self.filtered_signal = self.filtered_signal[512:]#cut away the settlement time of the filter
        
        signal_chunks=[]
        for i in range(0, len(self.filtered_signal), chunk_size):
            chunk = self.filtered_signal[i:i+chunk_size]
            if len(chunk) == chunk_size:
                signal_chunks.append(chunk)
        
        self.signal_chunks = signal_chunks    
        
class JoinSignals:
    '''
    The class for every action where both signals are needed. Includes:
        __init__
        synchronizing
        cut
    '''
    
    def __init__(self, signal_A, signal_B, timestamps_A, timestamps_B,):
        '''
        

        Parameters
        ----------
        signal_A : array
            The Referencesignal(Somno).
        signal_B : array
            The second signal (Corsano).
        timestamps_A : Series
            The reference timestamps as pandas.Series Objekt.
        timestamps_B : Series
            the second timestamps as pandas.Series Objekt.

        Returns
        -------
        None.

        '''
        self.signal_A = signal_A
        self.signal_B = signal_B
        self.timestamps_A = timestamps_A
        self.timestamps_B = timestamps_B
        
    def syncronizing(self, flag=0):
        '''
        

        Parameters
        ----------
        flag : bool, optional
            If True, indicates that no equal timestamps were found. Thus adding 12 hours to the reference timestamps
            anr reiterating in a recursive mannor. The default is 0.

        Returns
        -------
        self.sync: tupel
            The tupel with synchrinized indizes of somno, corsano and the true fs of corsano respectively.
                    
        '''
        som_flag = False
        cors_flag= False
        
        timestamps_B = self.timestamps_B
        
        timestamps_A = self.timestamps_A.dt.round('ms')
        timestamps_B = self.timestamps_B.dt.round('ms')
       
        #checking fs of corsano
        timedelta_fs = np.diff(timestamps_B)/np.timedelta64(1, 's')
        true_fs = 1/timedelta_fs
        check_fs = np.mean(true_fs)
        
        if (48 > check_fs):
            true_fs = 32
        
        if (48 <= check_fs):
            true_fs = 64
          
        # check timestamps    
        check_A = timestamps_A[timestamps_A.isin(timestamps_B)]# checks if a timestamp in somno is in corsano
        check_B = timestamps_B[timestamps_B.isin(timestamps_A)]#  checks if a timestamp in corsano is in somno

        check_A = check_A.reset_index()
        check_B = check_B.reset_index()
        
        if not check_A.empty: #if check_a is filled with something
            som_idx=check_A['index'][0] #take the first timestamp that checks out
            som_flag = True
        
        if not check_B.empty: ##if check_b is filled with something
            cors_idx = check_B['index'][0] #take the first timestamp that checks out
            cors_flag = True
         
        if som_flag and cors_flag: #if both checks are filled with something
            self.sync = (som_idx, cors_idx, true_fs) #return the tuple
            return 0
        
        if som_flag == False and cors_flag == False and flag == 0: # if both checks failed and 12h were not added
            self.timestamps_A += pd.Timedelta(hours = 12) #adding 12h  
            return self.syncronizing(flag=1) #recursion
        
    def cut(self, signal_length_A, signal_length_B):
        '''
       Cuts both signals to the same length.

        Parameters
        ----------
        length_A : Int, optional
            Desired length of the reference signal. The default is 138240 (18 min).
        length_B : Int, optional
            Desired length of the second signal. The default is 34560 (18 min).

        Returns
        -------
        None.

        '''
        signal_length_A+=(4 * fs_A)#adding settlement time of filter
        signal_length_B+=(4 * self.sync[2])#adding settlement time of filter
        idx_A, idx_B = self.sync[0], self.sync[1]
        while True:
            try:
                #checking if actual length is long enough. If not, iterative shortening until it fits.
                if len(self.signal_A) < signal_length_A or len(self.signal_B) < signal_length_B:
                    warnings.warn('Warning, signal_A or signal_B is too short. Targetlength will be shortened to fit the signallength.', UserWarning)
                    signal_length_A-=7680
                    signal_length_B-=int(60*self.sync[2])
        
                else:
                    self.signal_A = self.signal_A[idx_A: idx_A+signal_length_A]
                    self.signal_B = self.signal_B[idx_B: idx_B+signal_length_B]
                    
                    self.timestamps_A = self.timestamps_A[idx_A: idx_A+signal_length_A]
                    self.timestamps_B = self.timestamps_B[idx_B: idx_B+signal_length_B]
                    self.timestamps_A = self.timestamps_A.reset_index(drop=True)
                    self.timestamps_B = self.timestamps_B.reset_index(drop=True)
                    
                    break
        
            except Exception as e:
                print(f"An unexpected error occured: {e}")
                break
            
class SQI:
    '''
    The class to calculate all SQIs. Includes:
        __init__
        shanon_entropy
        calc_SNR
        ZCR
        skewness
        kurt
        decision (DEPRECATED)
        final_decision (DEPRECATED)
        
    '''
    
    def __init__(self, signal_chunks, criterion_snr=0, criterion_entropy=0, criterion_zcr=0):
        '''
        

        Parameters
        ----------
        signal_chunks : array
            A 2D-array with the sequences.
        criterion_snr : float, optional
            !!!Deprecated!!!
            Decision point for SNR. The default is 0.
        criterion_entropy : TYPE, optional
            !!!Deprecated!!!
            Decision point for entropy. The default is 0.
        criterion_zcr : TYPE, optional
            !!!Deprecated!!!    
            Decision point for ZCR. The default is 0.

        '''

        self.signal_chunks = signal_chunks
        self.criterion_snr = criterion_snr
        self.criterion_entropy = criterion_entropy
        self.criterion_zcr = criterion_zcr
        
    def shanon_entropy(self, num_bins=16):
        '''
        Calculates the entropy for every sequence.

        Parameters
        ----------
        num_bins : Int, optional
            The number of bins used. The default is 16.

        Returns
        -------
        self.entropy_values: list
            the entropy values for every sequence.

        '''
        entropy_values=[]
        for a in range(len(self.signal_chunks)):
            quantized_signal, bin_edges = np.histogram(self.signal_chunks[a], bins=num_bins, density=True)
            probabilities = quantized_signal / np.sum(quantized_signal)
            # Shannon-Entropie berechnen
            shannon_entropy = entropy(probabilities)/-np.log(1/num_bins)#normalised entropy by the value of equal probabilities
            entropy_values=np.append(entropy_values, shannon_entropy)
            
        self.entropy_values = entropy_values
        
    def calc_SNR(self, lowcut=low_BPM/60, highcut=high_BPM/60):
        '''
        Calculates the Signal- to Noise-Ratio.
        Defined as the relative power of a signalband to the rest.

        Parameters
        ----------
        lowcut : float, optional
            Lower bound of the frequency band. The default is .8 (48 bpm).
        highcut : float, optional
            higher bound of the frequency band. The default is 2.0 (120 bpm).

        Returns
        -------
        self.SNR: list
            The SNR-values for every sequence.
        self.freq: list
            the frequencies for every sequence.
        self.magnitude: list
            the magnitudes of the frequencies for every sequence.

        '''
        snr=[]
        frequencies = []
        magnitudes = []
        for i in range(len(self.signal_chunks)):
            freq = np.fft.fftfreq(len(self.signal_chunks[i]), d=1/fs_A)
            fft_values=np.fft.fft(self.signal_chunks[i])
            magnitude = np.abs(fft_values)
    
            signal_band = (freq >= lowcut) & (freq <= highcut)
            # signal_power = np.mean(np.sum(fft_values[signal_band])**2)
            signal_power = np.abs(np.trapz(fft_values[signal_band]))
            
            
            # noise_band = ~signal_band  # Frequenzbereich außerhalb der Signalbandbreite
            noise_band = (freq > highcut)  # Frequenzbereich außerhalb der Signalbandbreite
            # noise_band = (freq < lowcut)  # Frequenzbereich außerhalb der Signalbandbreite
            # noise_power = np.mean(np.sum(fft_values[noise_band])**2)
            noise_power = np.abs(np.trapz(fft_values[noise_band]))
            
            snr_db = 10 * np.log10(signal_power / noise_power)
            snr.append(snr_db)
            frequencies.append(freq)
            magnitudes.append(magnitude)
            
        self.SNR = (snr)
        self.freq = frequencies
        self.magnitude = magnitudes
            
    def ZCR(self):
        '''
        Calculate the ZCR for every sequence.

        Returns
        ------
        self.signs: list
            A list with values for  the signs of the signalvalues.
             1 = positive
            -1 = negative
        self.cross_pos: list
            A list if the indices where a zero crossing occured.
        self.n_cross: Int
            The number of zero crossings.
        
        self.slope: float
            the slope of the lin. regression for every sequence.
        self.intersect: float
            the the y-intersect of the lin. regression for every sequence.
        self.variance:
            the variance for every sequence.
        '''
        
        signs=[]
        cross_pos=[]
        n_cross=[]
        for i in range(len(self.signal_chunks)):
            tempsigns=np.sign(self.signal_chunks[i])
            crossings=np.diff(tempsigns)
            tempcrossing_positions=np.where(crossings != 0)[0] #only True where crossing[i] != crossing[i+1]
            temp_n_crossings=len(tempcrossing_positions)    
            
            signs.append(tempsigns)
            cross_pos.append(tempcrossing_positions)
            n_cross.append(temp_n_crossings)
        
        self.signs = signs
        self.cross_pos = cross_pos
        self.n_cross = n_cross

        self.slope, self.intersect = lin_reg(self.signal_chunks, cross_pos)
        self.variance = variance(self.slope, self.intersect, cross_pos)

    def skewness(self):
        '''
        Calculates the skewness for every sequence.

        Returns
        -------
        self.skewness: list
            The skewness for every sequence.

        '''
        
        skewness = []
        
        for chunk in self.signal_chunks:
            skewness.append(skew(chunk))
            
        self.skewness = skewness
        
    def kurt(self):
        '''
        Calculates the kurtosis for every sequence.

        Returns
        -------
        self.kurt: list
            The kurtosis for every sequence.

        '''

        
        kurt = []
        
        for chunk in self.signal_chunks:
            kurt.append(kurtosis(chunk))
            
        self.kurt = kurt
        
    def decision(self):
        '''
        !!!Deprecated!!!
        Decides wether the values are good or bad.

        Returns
        -------
        self.decision_snr: list
            List of bool. True for good value and Flase for bad value.
        self.decision_entropy: list
            List of bool. True for good value and Flase for bad value.
        self.decision_zcr: list
            List of bool. True for good value and Flase for bad value.

        '''
        self.decision_snr = decide(self.SNR, self.criterion_snr)
        self.decision_entropy = decide(self.entropy_values, self.criterion_entropy)
        self.decision_zcr = decide(self.variance, self.criterion_zcr, zcr=True)
    
    def final_decision(self):
        '''
        !!!Deprecated!!!
        A digital logic to combine all SQIs to one. Was used for SNR, Entropy and ZCR.
        Since these parameters dont apply its not used anymore.

        Returns
        -------
        wrapper : dict
            A dictionary with the final decision. It contains:
                min. Annehmbar: the length of chains of succesive sequences that are at least 'annehmbar'
                Gut: the length of chains of succesive sequences that are at least 'good' 
                längste min. Annehmbar in Prozent: the longes chain of at least 'annehmbar' in percent of the total amount of sequences
                längste Gut in Prozent: the longes chain of at least 'good' in percent of the total amount of sequences,
                Qualitäten: the raw qualities for every sequence

        '''
        
        qualities=np.array([])
        
        #logic to decide
        for i in range(len(self.decision_snr)):
            if self.decision_snr[i] == 1:
                if NOR(self.decision_entropy[i], self.decision_zcr[i]):
                    #print('Signalabschnitt', i, 'ist Schlecht')
                    qualities = np.append(qualities, 1)
                elif XOR(self.decision_entropy[i], self.decision_zcr[i]):
                    #print('Signalabschnitt', i, 'ist Annehmbar')
                    qualities = np.append(qualities, 2)
                elif self.decision_entropy[i] and self.decision_zcr[i] == 1:
                    #print('Signalabschnitt', i, 'ist Gut')
                    qualities = np.append(qualities, 3)        
            else:
                #print('Signalabschnitt', i, ' ist Unbrauchbar ')
                qualities = np.append(qualities, 0)
        
        mask = np.isin(qualities, [2, 3])  # Mask: True for 2 & 3 -> at least 'annehmbar'
        chains=[]
        count=0
        for value in mask:
            if value == True:
                count+=1
            else:
                if count > 0:  # append only if chain ist > 0
                    chains.append(count)
                count=0
        
        if count > 0: 
            chains.append(count)
            count=0
        
        if len(chains) != 0:
            max_chain=max(chains) 
            percentage_OK=(max_chain/len(qualities))*100
        
        else:
            max_chain=0
            percentage_OK=0
        
        mask = np.isin(qualities, 3)  # Mask: True for 3 -> at least 'good'
        good_chains=[]
        for value in mask:
            if value == True:
                count+=1
            else:
                if count > 0:  # Nur hinzufügen, wenn die Kette mehr als 0 ist
                    good_chains.append(count)
                count=0
        
        if count > 0: 
            good_chains.append(count)
        
        if len(good_chains) != 0:
            max_good_chain=max(good_chains)
            percentage_good=(max_good_chain/len(qualities))*100
        else:
            max_good_chain=0
            percentage_good=0
            
        
        wrapper=dict({'min. Annehmbar': chains,
                'Gut': good_chains,
                'längste min. Annehmbar in Prozent': percentage_OK,
                'längste Gut in Prozent': percentage_good,
                'Qualitäten': qualities
                })
        
        return wrapper

#training bib


class Training:
    '''
    The class for Training. Includes:
        __init__
        separate_values
        building hists
    
    '''
    def __init__(self, skew_values, kurt_values, data):
        '''
        

        Parameters
        ----------
        skew_values : array
            The array with all skewness values.
        kurt_values : array
            The array with all kurtosis values.
        data : array
            The annotation.


        '''
        
        self.skew_values = np.asarray(skew_values)
        self.kurt_values = np.asarray(kurt_values)
        self.data = data
        
    def separate_values(self):
        '''
        Separating the values in good and bad data according to the annotation.

        '''
        
        if len(self.data)-len(self.skew_values) == 0: #check for equal length -> human error
            positive_idx = np.where(np.asarray(self.data) == 1)[0]
            negative_idx = np.where(np.asarray(self.data) == 0)[0]
    
            #sorting the values with respect to the annotation
            self.positive_skew = self.skew_values[positive_idx]
            self.positive_kurt = self.kurt_values[positive_idx]
            
            self.negative_skew = self.skew_values[negative_idx]
            self.negative_kurt = self.kurt_values[negative_idx]
        
        else:
            print('Annotation and Signal are not of the same lenght!')
            print(f'Annotation {len(self.data)}, Signal {len(self.skew_values)}')
            
    def building_hists(self, plot=False):
        '''
        

        Parameters
        ----------
        plot : bool, optional
            If True, all histogrammst will be plotted. The default is False.

        Returns
        -------
        self.hist: aray
            The histogrammvalues.
        self.xedges: array
            The xedges if the bins.
        self.yedges: array            
            The yedges if the bins.

        '''
            
        bins = 1000
        range_xy=[[-3, 3], [-3, 3]]
        
        hist_good, xedges, yedges = np.histogram2d(self.positive_skew, self.positive_kurt, bins=bins, range=range_xy, density=True)
        hist_bad, _, _= np.histogram2d(self.negative_skew, self.negative_kurt, bins=bins, range=range_xy, density=True)
        
        #normalising hists
        hist_good = hist_good/hist_good.sum()
        hist_bad = hist_bad/hist_bad.sum()
        
        hist_diff = np.log(hist_good + 1e-10) - np.log(hist_bad + 1e-10)
      
        hist_smoothed = gaussian_filter(hist_diff, sigma=1.5)  # sigma kannst du anpassen
        
        self.hist = hist_smoothed
        self.xedges = xedges
        self.yedges = yedges
        
        if plot:
            
            cmap = 'afmhot'
            fig, axes= plt.subplots(ncols=2, nrows=2, figsize=(18,18))
            ax1, ax2, ax3, ax4 = axes.ravel() 
            
            extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
            im = ax1.imshow(hist_good.T, origin='lower', extent=extent, aspect='auto', cmap=cmap)
            cbar1 = plt.colorbar(im, ax=ax1, label='Dichte gut')
            cbar1.set_label('Dichte gut')  
            ax1.set_xlabel('Skewness')
            ax1.set_ylabel('Kurtosis')
            
            im = ax2.imshow(hist_bad.T, origin='lower', extent=extent, aspect='auto', cmap=cmap)
            cbar2 = plt.colorbar(im, ax=ax2, label='Dichte schlecht')
            cbar2.set_label('Dichte schlecht')  
            ax2.set_xlabel('Skewness')
            ax2.set_ylabel('Kurtosis')
           
        
            im = ax3.imshow(hist_diff.T, origin='lower', extent=extent, aspect='auto', cmap=cmap)
            cbar3 = plt.colorbar(im, ax=ax3, label='log-Dichtdifferenz')
            cbar3.set_label('log-Dichtedifferenz')
            ax3.set_xlabel('Skewness')
            ax3.set_ylabel('Kurtosis')
            
            im = ax4.imshow(hist_smoothed.T, origin='lower', extent=extent, aspect='auto', cmap=cmap)
            cbar4 = plt.colorbar(im, ax=ax4)
            cbar4.set_label('log-Dichtedifferenz geglättet')
            ax4.set_xlabel('Skewness')
            ax4.set_ylabel('Kurtosis')
            
            ax1.grid(False)
            ax2.grid(False)
            ax3.grid(False)
            ax4.grid(False)
            
            plt.tight_layout()
            plt.show()

def classify_data(hist, data1, data2, xedges, yedges, annotation=None, plot=False):
    '''
    Classify the input data.

    Parameters
    ----------
    hist : array
        The histogramvalues.
    data1 : array
        The skewness values of a signal.
    data2 : array
        The kurtosis values of a signal.
    xedges : array
        The xedges of the bins.
    yedges : array
        The yedges of the bins.
    annotation : array, optional
        The annotation. It is only used to plot the data into the histogramm. The default is None.
    plot : bool, optional
        If True, the histogram will be plotted with the annotated data. The default is False.

    Returns
    -------
    scores : array
        The likelihood of a value being good.

    '''
    
    scores = []
    #lokating the skew and kurtvalues in the histogram
    for x_val, y_val in zip(data1, data2):
        x_idx = np.searchsorted(xedges, x_val, side="right") - 1
        y_idx = np.searchsorted(yedges, y_val, side="right") - 1
        # z_idx = np.searchsorted(zedges, z_val, side="right") - 1
        
        # checking for the value being within the boundaries
        if (0 <= x_idx < hist.shape[0] and 0 <= y_idx < hist.shape[1]):
        
            score = hist[x_idx, y_idx]
        else:
            score= 0
            
        scores.append(score)
    
    
    if plot:
        good_idx = np.where(annotation==1)
        good1 = data1[good_idx]
        good2 = data2[good_idx]
        
        bad_idx = np.where(annotation==0)
        bad1 = data1[bad_idx]
        bad2 = data2[bad_idx]
        
        
        xcenters = 0.5 * (xedges[:-1] + xedges[1:])
        ycenters = 0.5 * (yedges[:-1] + yedges[1:])
        X, Y = np.meshgrid(xcenters, ycenters)
        
        # Jetzt korrekt plotten
        plt.figure(figsize=(8, 6))
        plt.contourf(X, Y, hist.T, levels=100, cmap="viridis")
        plt.scatter(good1, good2, color='b')
        plt.scatter(bad1, bad2, color='r')
        plt.colorbar(label="Glättete Häufigkeit")
        plt.xlabel("X (original scale)")
        plt.ylabel("Y (original scale)")
        plt.title("Geglättetes 2D-Histogramm mit korrekter Skalierung")
        plt.tight_layout()
        plt.show()
        
    
    return scores

def train_hist(validation_sets, hists, plot=False):
    '''
    This funktion iterates over the subsets. Here the thresholds for every subset are determined with the use of ROC.

    Parameters
    ----------
    validation_sets : dict
        A dictionaity containing the data of the subsets -> import_training_data().
    hists : dict
        A dictionairy containing the histograms of every trainingset.
    plot : bool, optional
        If True, all five ROC-Curves are plotted. The default is False.

    Returns
    -------
    master_thresholds : dict
        A dictionairy containing the beste threshold for every subset.

    '''

    master_thresholds = defaultdict(dict)
    master_thresholds_array = defaultdict(dict)
    master_fpr = defaultdict(dict)
    master_tpr = defaultdict(dict)
    master_idx = defaultdict(dict)
    master_auc = defaultdict(dict)
    
    for subset in validation_sets:
        
        data = validation_sets[subset]
        data_skew = data[:,0]
        data_kurt = data[:,1]
        global data_anno #debugg
        data_anno = data[:,5].astype(bool)
        scoring_hist = hists[subset][0]
        scoring_xedges = hists[subset][1]
        scoring_yedges = hists[subset][2]
        global scores #debugg
        #scoring
        scores = classify_data(scoring_hist, data_skew, data_kurt, scoring_xedges, scoring_yedges, data_anno, plot=False)
        
        #calc parameters
        fpr, tpr, thresholds_roc = roc_curve(data_anno, scores)
        roc_auc = auc(fpr, tpr)
        
        # youden_j = tpr - fpr
        # best_idx = youden_j.argmax()
        # best_threshold = thresholds[best_idx]
        
        precision, recall, thresholds_pr = precision_recall_curve(data_anno, scores)
        # avg_prec = average_precision_score(data_anno, scores)

        #calc best threshold with f_beta score
        beta = .12 # z.B. für FP vermeiden
        f_beta_scores = (1 + beta**2) * precision * recall / (beta**2 * precision + recall + 1e-10)
        best_idx = np.argmax(f_beta_scores)
        best_threshold = thresholds_pr[best_idx]
        
        master_thresholds[subset] = np.mean(best_threshold)

        #preparing plot
        plot_idx = np.where(best_threshold == thresholds_roc)[0]
                
        master_fpr[subset] = fpr
        master_tpr[subset] = tpr
        master_thresholds_array[subset] = thresholds_pr
        master_idx[subset]=plot_idx 
        master_auc[subset]=roc_auc
   
    if  plot:
        
        fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18,18))
        ax = axes.ravel() 

        for i, subset in enumerate(master_fpr):
            best_idx = master_idx[subset]
            ax[i].plot(master_fpr[subset], master_tpr[subset], color='tab:orange', lw=2, label=f'ROC curve (AUC = {master_auc[subset]:.2f})')
            ax[i].scatter(master_fpr[subset][best_idx], master_tpr[subset][best_idx], color='tab:blue')

            ax[i].plot([0, 1], [0, 1], color='k', lw=2, linestyle='--')  # Diagonale als Referenz
            ax[i].set_xlim([0.0, 1.0])
            ax[i].set_ylim([0.0, 1.05])
            ax[i].set_xlabel('False Positive Rate')
            ax[i].set_ylabel('True Positive Rate')
            ax[i].set_title(f'Fold {i}')
            ax[i].legend(loc="lower right")
        ax[5].axis('off')
        plt.tight_layout()    
        plt.show()
            
            
    return master_thresholds
      
def mean_hists(hists, thresholds):
    '''
    Calculates the mean from all histograms as final classifier.

    Parameters
    ----------
    hists : dict
        All five histogramms fromm the trainingsets.
    thresholds : dict
        best threshold from every subset.

    Returns
    -------
    mean_hist : array
        The mean histogram.        
    xedges : array
        The xedges of the bins.
    yedges : array
        The yedges of the bins.
    mean_thresh : float
        the mean threshold from the best thresholds.

    '''
   
    all_hists = []
    all_thresh = []
    for subset in hists:
        hist = hists[subset][0]
        thresh = thresholds[subset]
        all_thresh.append(thresh)
        all_hists.append(hist)
    
    mean_hist = np.mean(all_hists, axis=0) 
    mean_thresh = np.mean(all_thresh)
    xedges = hists['subset 0'][1] # since the bins and boundaries of all hist are the same
    yedges = hists['subset 0'][2] # since the bins and boundaries of all hist are the same
    # zedges = master_hists['subset 0'][3]
    
    del all_hists # saving memory
        
    return mean_hist, xedges, yedges, mean_thresh
    
def test_subsets(validation_sets, mean_hist, xedges, yedges, best_thresh, plot=False):
    '''
    

    Parameters
    ----------
    validation_sets : dict
        A dictionaity containing the data of the subsets -> import_training_data().
    mean_hist : array
        The classifying histogram.
    xedges : array
        The xedges of the bins.
    yedges : array
        The yedges of the bins.
    best_thresh : float
        The mean threshold.
    plot : TYPE, optional
        If True, the performance of the calssifier on the subsets will be plotted in ROC-Space.
        The default is False.

    Returns
    -------
    tuple
        A tuple containing the 
        mean fpr
        mean tpr
        standard deviation of fpr
        standard deviation of tpr
        on the respective indices.

    '''
    
    labels=[0,1]# initializing labels for the confusion matrix
    master_analytics = defaultdict(dict)

    all_fpr = []
    all_tpr = []

    for subset in validation_sets:
        
        data = validation_sets[subset]
        data_skew = data[:,0]
        data_kurt = data[:,1]
        
        data_anno = data[:,5]        

        #scoring
        scores = classify_data(mean_hist, data_skew, data_kurt,
                               xedges, yedges)
       
        #predicting
        y_pred = (scores >= best_thresh).astype(int)
            
        #confusion matrix
        cm = confusion_matrix(data_anno, y_pred, labels=labels)
         
        #calc parameters            
        fpr = cm[0,1]/np.sum(cm[0]) if np.sum(cm[0]) > 0 else np.nan
        tpr = cm[1,1]/np.sum(cm[1]) if np.sum(cm[1]) > 0 else np.nan
            
        all_fpr.append(fpr)
        all_tpr.append(tpr)
        
        master_analytics[subset] = {'fpr': fpr, 'tpr': tpr}
         
    mean_fpr = np.nanmean(all_fpr)
    mean_tpr = np.nanmean(all_tpr)

    print('all tpr', all_tpr)
    print('all fpr', all_fpr)
    print('mean tpr', mean_tpr)
    print('mean fpr', mean_fpr)
 
    if plot:
        plt.figure()
        for subset in master_analytics:
            fpr = master_analytics[subset]['fpr']
            tpr = master_analytics[subset]['tpr']
            
            plt.scatter(fpr, tpr, label=subset, alpha=.5)
    
        plt.scatter(mean_fpr, mean_tpr, color ='r', label='mean')
        plt.plot([0,1], [0,1], color='k', linestyle='--')
    
        plt.legend(loc='lower right')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
       
        plt.xlim(0,1)
        plt.ylim(0,1)
        plt.tight_layout()
        plt.show()
        

    return (mean_fpr, mean_tpr, np.std(all_fpr), np.std(all_tpr))

def test_testset(test_set, mean_hist, xedges, yedges, best_thresh, val_tupel, plot=False):
    '''

    Parameters
    ----------
    test_set : ndarray
        A dictionaity containing the data of the testset -> import_training_data().
    mean_hist : array
        The classifying histogram.
    xedges : array
        The xedges of the bins.
    yedges : array
        The yedges of the bins.
    best_thresh : float
        The mean threshold.
    val_tupel : tuple
        A tuple containing the 
            mean fpr
            mean tpr
            standard deviation of fpr
            standard deviation of tpr
        on the respective indices.
    plot : bool, optional
        If True, the performance ofe the classifier on the testset will be plottet in the ROC-space
        with the data of the subsets as errorbars. The default is False.

    Returns
    -------
    test_fpr : float
        The fpr of the classifier on the testset.
    test_tpr : float
        The tpr of the classifier on the testset.

    '''
    #label for confusion matrix
    labels = [0,1]
    
    data = test_set
    data_skew = data[:,0]
    data_kurt = data[:,1]
   
    data_anno = data[:,5]     
        
    reference = list(map(int, data_anno))
    
    #scoring    
    scores = classify_data(mean_hist, data_skew, data_kurt,
                           xedges, yedges, data_anno, plot=False)
    
    #prediction
    y_pred = (scores >= best_thresh).astype(int)
    #confusion matrix    
    cm = confusion_matrix(reference, y_pred, labels=labels)
    #calc parameters
    test_fpr = cm[0,1]/np.sum(cm[0]) if np.sum(cm[0]) > 0 else np.nan
    test_tpr = cm[1,1]/np.sum(cm[1]) if np.sum(cm[1]) > 0 else np.nan

    if plot:
        plt.figure()
        plt.scatter(test_fpr, test_tpr, color ='tab:red', label=f'TPR = {test_tpr: .3f}'+'\n'+
                    f'FPR = {test_fpr: .3f}')
        plt.errorbar(val_tupel[0], val_tupel[1], xerr=val_tupel[2], yerr=val_tupel[3], color='tab:green', capsize=5, label=rf'Hold-Out TPR = {val_tupel[1]: .3f}$\pm${val_tupel[3]: .3f}' + '\n'+
                     rf'Hold-Out FPR = {val_tupel[0]: .3f}$\pm${val_tupel[2]: .3f}')
        plt.plot([0,1], [0,1], color='k', linestyle='--')
        
        plt.legend()
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        
        plt.xlim(0,1)
        plt.ylim(0,1)
        plt.tight_layout()
        plt.show()
        
    return test_fpr, test_tpr

def wrapping_results(prediction):
    '''
    Counting the good and bad labeled sequences.

    Parameters
    ----------
    prediction : array
        The array with the prdicted labels. 

    Returns
    -------
    tuple
        A tuple containing the number of good labels, bad labels and overall number of labels.

    '''
    
    good = np.where(prediction == 1)[0]
    bad = np.where(prediction == 0)[0]
    
    return len(good), len(bad), len(good)+len(bad)

      
#%% pipeline
TN_list = os.listdir(ordner_pfad) # geting the names

def process(TN_list, train=False,  plot=False):
    '''
    

    Parameters
    ----------
    TN_list : list
        A list of the names.
    train : bool, optional
        If True, PPG-Eva is in training mode. Otherwise it is in Evaluation mode. The default is False.
    plot : bool, optional
        Deprecated. The default is False.

    Returns
    -------
    in Train-mode:
        Nothing. The results of the classifier are shown. The operator is prompted to enter:
         y : classifier will be saved and PPG-Eva terminates.
         n : training will be performed again with a new randomization of the sets.
         q : Quit. PPG.Eva terminates.

    in Evaluation-mode:
        final_results_somno: dict
            A dictionairy where the names are the keys. Behind every key is a dictionairy containing the results from 
            wrapping_results()
            'good': number og good sequences
            'bad' : number of bad sequences
            'sum' : nuber of sequences

    '''
            
    def training():
        '''
        The trining Pipeline. It uses the extracted values from the Reference dataset to build a 
        2D-histogram based calssifier.

        
        '''
        # all_fpr = []
        # all_tpr = []
        # for i in range(0,1000):
        #     print('Cycle', i)
        global training_sets, validation_sets, test_set #debugg
        training_sets, validation_sets, test_set = import_training_data(training_values_path)
  
        #building hist
        master_hists = defaultdict(dict)
        
        for subset in training_sets:
            #building hist for every subset
            good = training_sets[subset]['good']
            bad = training_sets[subset]['bad']
            
            init_values=np.concatenate((good, bad), axis=0)
            skew_values = init_values[:,0]
            kurt_values = init_values[:,1]
            anno_values = init_values[:,5]#entropy is skipped
           
            obj_training = Training(skew_values, kurt_values, anno_values)
            obj_training.separate_values()
            obj_training.building_hists(plot=False)
            
            master_hists[subset] = (obj_training.hist, obj_training.xedges, obj_training.yedges)

        #calc thresholds for every subset
        master_thresholds = train_hist(validation_sets, master_hists, plot=False)
        #mean hists            
        mean_hist, xedges, yedges, best_thresh = mean_hists(master_hists, master_thresholds)
        #performance test
        analysis_tupel = test_subsets(validation_sets, mean_hist, xedges, yedges, best_thresh, plot=True)
        
        test_fpr, test_tpr = test_testset(test_set, mean_hist, xedges, yedges, best_thresh, analysis_tupel, plot=True)
        #wrapping data for export        
        hist = {'counts': mean_hist, 'xedges': xedges, 'yedges': yedges, 'threshold': best_thresh}
        plt.pause(1)
            # all_fpr.append(test_fpr)
            # all_tpr.append(test_tpr)
            
            # if i == 5:
            #     break
        # userinput: y: sve classifier; n: start over; q: quit
        save = input('Save? (y/n/q):')
        
        if save == 'y':
            with open(r'F:\D\classifier.pkl', 'wb') as f:
                pickle.dump(hist, f)
        elif save =='q':
            return (0,0)
        elif save == 'n':
            process(TN_list, train=True)
             
    def no_train():
        '''
        Evaluation-mode
        Evaluates Signals based on the classifier. Both Signals are processed somewhat parallel.

        Returns
        -------
        final_result_somno : dict
            A dictionairy where the names are the keys. Behind every key is a dictionairy containing the results from 
            wrapping_results()
            'good': number og good sequences
            'bad' : number of bad sequences
            'sum' : nuber of sequences

        final_result_corsano : dict
            A dictionairy where the names are the keys. Behind every key is a dictionairy containing the results from 
            wrapping_results()
            'good': number og good sequences
            'bad' : number of bad sequences
            'sum' : nuber of sequences


        '''
        # import classifier
        with open(r'F:\D\classifier.pkl', 'rb') as f:
            hist = pickle.load(f)
           

        scoring_hist = hist['counts'] 
        xedges = hist['xedges']
        yedges = hist['yedges']
        thresh = hist['threshold']
       
        final_result_somno = defaultdict(dict)
        final_result_corsano = defaultdict(list)
      
        # importing a resolution table wich connects two datastructures
        df_auflösung = pd.read_excel(r'F:\ID_zu_messung.xlsx')
        df_auflösung=df_auflösung.set_index('Unnamed: 0')

        auflösung = defaultdict(dict, df_auflösung.to_dict(orient='index'))

        for key, messung in auflösung.items():
            auflösung[key] = messung[0]
        
        #the actual processing
        for name in TN_list:
            print('Processing', name)
            
            preprocessing(ordner_pfad, name, chunk_length=1/6, plot=False, training=False)
            #scoring
            scores_som = classify_data(scoring_hist, somno_SQI.skewness, somno_SQI.kurt, xedges, yedges)
            scores_cor = classify_data(scoring_hist, corsano_SQI.skewness, corsano_SQI.kurt, xedges, yedges)
            #predicting            
            y_pred_som = (scores_som >= thresh).astype(int)
            y_pred_cor = (scores_cor >= thresh).astype(int)
            #wrapping
            good_chunks_som, bad_chunks_som, sum_som = wrapping_results(y_pred_som)
            good_chunks_cor, bad_chunks_cor, sum_cor = wrapping_results(y_pred_cor)
            
            final_result_somno[name]={'good': good_chunks_som, 'bad': bad_chunks_som, 'sum': sum_som}
            final_result_corsano[name]={'good': good_chunks_cor, 'bad': bad_chunks_cor, 'sum': sum_cor}
            
            # break
        return final_result_somno, final_result_corsano
    
    if train:
        controll_fpr, controll_tpr = training()
        return (controll_fpr, controll_tpr)
    else:
        final_result_somno, final_result_corsano = no_train()
        return final_result_somno, final_result_corsano

        
result_somno, result_corsano = process(TN_list, train=True)
# result_somno, result_corsano = process(TN_list, train=False)

#%%

# all_zero = []
# all_one = []

# for subset in validation_sets:
#     zero = np.where(validation_sets[subset][:,5] == 0)[0]
#     one = np.where(validation_sets[subset][:,5] == 1)[0]
    
#     all_zero.append(len(zero))
#     all_one.append(len(one))
    
# print(np.sum(all_zero))
# print(np.sum(all_one))

# #%%

# zero = np.where(test_set[:,5] == 0)[0]
# one = np.where(test_set[:,5] == 1)[0]
    
    
# print(len(zero))
# print(len(one))















