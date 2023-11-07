import numpy as np
import librosa
from torch.utils.data import Dataset
from RawBoost import  process_Rawboost_feature			




class Dataset_train(Dataset):
    def __init__(self, args, list_IDs, labels, base_dir, algo):
        self.list_IDs = list_IDs
        self.labels = labels
        self.base_dir = base_dir
        self.algo=algo
        self.args=args   
    def __len__(self):
        return len(self.list_IDs)
    def __getitem__(self, index):
        utt_id = self.list_IDs[index]
        X,fs = librosa.load(self.base_dir+'wav/'+utt_id+'.wav', sr=16000) 
        Y=process_Rawboost_feature(X,fs,self.args,self.algo)
        X_rs = np.reshape(Y,(1,-1))
        target = self.labels[utt_id]
        return X_rs, target, None

class Dataset_eval(Dataset):
    def __init__(self, list_IDs, base_dir, track):
        '''self.list_IDs	: list of strings (each string: utt key),'''
        self.list_IDs = list_IDs
        self.base_dir = base_dir
        self.track=track
    def __len__(self):
        return len(self.list_IDs)
    def __getitem__(self, index):  
            utt_id = self.list_IDs[index]
            X, fs = librosa.load(self.base_dir+'wav/'+utt_id+'.wav', sr=16000) 
            X_rs = np.reshape(X,(1,-1))
            return X_rs, None, utt_id  
    
