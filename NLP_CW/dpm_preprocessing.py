from hashlib import new
from dont_patronize_me import DontPatronizeMe
from dpm_preprocessing_utils import apply_preprocessing
import pandas as pd
from sklearn.model_selection import train_test_split
#from data_augmentation import augment_substitute_with_context

class DPMProprocessed(DontPatronizeMe):
    def __init__(self, train_path, test_path):
        super().__init__(train_path, test_path)
        self.load_task1()
        self.load_task2(return_one_hot=True)
        self.load_test()

        self._preprocess_all_df()

        self.train_task1_df['lenght'] = self.train_task1_df['text'].apply(lambda s: len(s.split()))

        self.positive_samples = self.train_task1_df[self.train_task1_df['label'] == 1 ]
        self.negative_samples = self.train_task1_df[self.train_task1_df['label'] == 0 ]
        print(self.train_task1_df)

    
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

    def get_unbalanced_split(self, val_size = 0.2):
        
        return train_test_split(self.train_task1_df, test_size=val_size, stratify=self.train_task1_df['label'])

    def get_oversampled_split(self, oversampling_ratio=10, val_size = 0.2):
        train_df, val_df = self.get_unbalanced_split(val_size)
        train_df_pos = train_df[train_df['label'] == 1]

        to_concat = [train_df]

        for _ in range(oversampling_ratio - 1):
            new_sampled_df = train_df_pos.copy()
            new_sampled_df['text'] = new_sampled_df['text'].apply(augment_substitute_with_context)
        
            to_concat.append(new_sampled_df)

        train_df = pd.concat(to_concat)

        return train_df, val_df
        







        

        


                
    

    

    

    
