from dont_patronize_me import DontPatronizeMe
from dpm_preprocessing_utils import apply_preprocessing
import pandas as pd
from sklearn.model_selection import train_test_split

class DPMProprocessed(DontPatronizeMe):
    def __init__(self, train_path, test_path):
        super().__init__(train_path, test_path)
        self.load_task1()
        self.load_task2(return_one_hot=True)
        self.load_test()

        self._preprocess_all_df()

        self.train_task1_df['lenght'] = self.train_task1_df['text'].apply(lambda s: len(s.split()))

        self.positive_samples = self.train_task1_df[self.train_task1_df['label'] == 1]
        self.negative_samples = self.train_task1_df[self.train_task1_df['label'] == 0]

    
    def _preprocess_all_df(self):
        for df in (self.train_task1_df, self.train_task2_df, self.test_set_df):
            df['text'] = df['text'].apply(apply_preprocessing)

    def get_downsampled_split(self, ratio = 1, val_size = 0.2):
        #ratio between number of pos and neg
        n_neg = len(self.negative_samples)
        n_pos = len(self.positive_samples)

        n_neg_final = n_pos * ratio
        assert(n_neg_final <= n_pos)

        negative_downsampled_samples = self.negative_samples.sample(frac=n_neg_final/n_neg)
        
        df = pd.concat([negative_downsampled_samples, self.positive_samples])
        
        return train_test_split(df, test_size=val_size, stratify=df['label'])





        

        


                
    

    

    

    
