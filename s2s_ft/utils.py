from __future__ import absolute_import, division, print_function
import logging
import os
import json
import random
import glob
import re

import torch
import tqdm
import torch.utils.data
class Seq2seqDatasetForLayoutlm_self(torch.utils.data.Dataset):
    def __init__(self, features,num_training_instances,with_label=False):
        self.features = features
        self.offset = 0
        self.num_training_instances = num_training_instances
        self.with_label=with_label

    def __len__(self):
        return int(self.num_training_instances)

    def __getitem__(self, idx):
        return self.__getitem_layout___self(idx)

    def __getitem_layout___self(self, idx):
        idx = (self.offset + idx) % len(self.features)
        feature = self.features[idx]
        input_sentence_coord=feature["input_sentence_coord"]
        target_index = feature['target_index']
        sentence_num=feature['sentences_num']
        if self.with_label:
            input_label_data=feature['label_data'] 
            return input_sentence_coord,target_index,sentence_num,input_label_data
        else:
            return input_sentence_coord,target_index,sentence_num
        
class TextLineDataset(torch.utils.data.Dataset):
    def __init__(self, features,with_label):
        self.features = features
        self.num_instances = len(features)
        self.with_label=with_label
        
    def __len__(self):
        return int(self.num_instances)

    def __getitem__(self, index):
        return self.__getitem_layout___self(index)
    
    def __getitem_layout___self(self, index):
        index = (index) % len(self.features)
        feature = self.features[index]
        input_ids=feature["input_ids"]
        sentences_num=feature["sentences_num"]
        tgt_index = feature['tgt_index']
        input_label_data=feature['input_label_data'] 
        return input_ids,sentences_num,tgt_index,input_label_data      
         
def batch_list_to_batch_tensors_self(batch):
    batch_tensors = []
    for x in zip(*batch):
        if isinstance(x[0], torch.Tensor):
            batch_tensors.append(torch.stack(x))
        else:
            batch_tensors.append(torch.tensor(x, dtype=torch.long))
    return batch_tensors

def load_and_cache_only_layout(example_path,cached_features_file,with_label=False,shuffle=False):
    if cached_features_file is not None and os.path.exists(cached_features_file):
        print("Loading features from cached file ", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        print("Creating features from dataset at ", example_path)
        examples = []
        if os.path.isdir(example_path):
            text_files = glob.glob(f'{example_path}/*layout*.json')
            layout_files = [re.sub('text|txt', 'layout', x, 1) for x in text_files]
        for layout_file in layout_files:
            with open(layout_file, mode='r', encoding='utf-8') as layout_reader:
                for i,layout_line in enumerate(layout_reader):
                    if (i + 1) % 10000 == 0:
                        print(f'{i + 1} lines ...')
                    examples.append((json.loads(layout_line)))
        features = []  
        def get_input_data(layout):
            """ sentence  num """
            sentence_num=layout["len:"]
            src_coord=layout["src"]
            """ coord """
            src_coord.insert(0,[0,0,0,0])
            src_coord.append([1000,1000,1000,1000])
            tgt_coord=layout["tgt"]
            pse_coord=[]
            """ tgt index """
            tgt_index=layout["tgt_index"]
            tgt_index.extend([0])
            if with_label:
                """ label """
                src_label=layout["src_label_note"]
                src_label.insert(0,0)
                src_label.append(6)
                tgt_label=layout["tgt_label_note"]
                pse_label=[]
                for tgt_coord_,tgt_label_ in zip(tgt_coord,tgt_label):
                    p=random.random()
                    if p < 0.1:
                        pse_coord.append(tgt_coord_)
                        pse_label.append(tgt_label_)
                    elif p < 0.2:
                        pse_coord.append([0,0,0,0])
                        pse_label.append(random.randint(0, 9))
                    else:
                        pse_coord.append([0,0,0,0])
                        pse_label.append(0) 
            else:           
                for tgt_coord_ in  tgt_coord:
                        p=random.random()
                        if p < 0.1:
                            pse_coord.append(tgt_coord_) 
                        elif p < 0.2:
                            pse_coord.append([0,0,0,0])
                        else:
                            pse_coord.append([0,0,0,0])       
            tgt_coord.append([1000,1000,1000,1000])   
            pse_coord.append([1000,1000,1000,1000])  
            if with_label:        
                tgt_label.append(6)
                pse_label.append(6)       
            """ padding """   
            if 513> len(src_coord):             #add pads
                    for j in range(( 513 - len(src_coord))):
                        src_coord.append([0,0,0,0])
                        if with_label:
                            src_label.append(0)
            if 511> len(tgt_coord):             #add pads
                    for j in range(( 511 - len(tgt_coord))):
                        tgt_coord.append([0,0,0,0])
                        pse_coord.append([0,0,0,0])
                        tgt_index.extend([0])
                        if with_label:
                            tgt_label.append(0)
                            pse_label.append(0)
                  
            line_coord_data= src_coord+tgt_coord+pse_coord
            if with_label:
                label_data=src_label+tgt_label+pse_label        
                return line_coord_data,sentence_num+1,tgt_index,label_data
            else:
                return line_coord_data,sentence_num+1,tgt_index
            
        for layout in tqdm.tqdm(examples):
            if with_label:
                input_sentence_coord_,sentence_num_,tgt_index_,label_data_=get_input_data(layout)
                input_label_data=torch.tensor(label_data_,dtype=torch.long)
            else:
                input_sentence_coord_,sentence_num_,tgt_index_=get_input_data(layout)
            input_sentence_coord=torch.tensor(input_sentence_coord_,dtype=torch.long)
            """ judge """
            judge_coord=input_sentence_coord[:, -1] - input_sentence_coord[ :,-3]
            judge_coord_=input_sentence_coord[:, 2] - input_sentence_coord[:, 0]
            if torch.lt(judge_coord_, 0).any()==True or torch.lt(judge_coord, 0).any():
                continue
            tgt_indexs=torch.tensor(tgt_index_,dtype=torch.long)
            sentence_num=torch.tensor(sentence_num_,dtype=torch.long)
            if sentence_num>510:
                continue
            if with_label:
                feature = {
                    "input_sentence_coord": input_sentence_coord,
                    "target_index": tgt_indexs,
                    "sentences_num":sentence_num,
                    "label_data":input_label_data,
                }
            else:
                feature = {
                    "input_sentence_coord": input_sentence_coord,
                    "target_index": tgt_indexs,
                    "sentences_num":sentence_num,
                }
                
            features.append(feature)
        if shuffle:
            random.shuffle(features)
        if not os.path.exists(os.path.dirname(cached_features_file)):
            os.makedirs(os.path.dirname(cached_features_file))
        print("Saving features into cached file ", cached_features_file)
        torch.save(features, cached_features_file)
    return features

def load_and_cache_box_label(example_path,cached_features_file,with_label=False,shuffle=False):
    if cached_features_file is not None and os.path.exists(cached_features_file):
        print("Loading features from cached file ", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        print("Creating features from dataset at ", example_path)
        examples = []
        if os.path.isdir(example_path):
            text_files = glob.glob(f'{example_path}/*layout*.json')
            layout_files = [re.sub('text|txt', 'layout', x, 1) for x in text_files]
        for layout_file in layout_files:
            with open(layout_file, mode='r', encoding='utf-8') as layout_reader:
                for i,layout_line in enumerate(layout_reader):
                    if (i + 1) % 10000 == 0:
                        print(f'{i + 1} lines ...')
                    examples.append((json.loads(layout_line)))
        features = []  
        def get_input_data(layout):
            """ sentence  num """
            sentence_num=layout["len:"]
            src_coord=layout["src"]
            """ coord """
            src_coord.insert(0,[0,0,0,0,0,0,0,0])
            src_coord.append([1000,1000,1000,1000,1000,1000,1000,1000])
            tgt_coord=layout["tgt"]
            pse_coord=[]
            """ tgt index """
            tgt_index=layout["tgt_index"]
            tgt_index.extend([0])
            if with_label:
                """ label """
                src_label=layout["src_label_note"]
                src_label.insert(0,0)
                src_label.append(6)
                tgt_label=layout["tgt_label_note"]
                pse_label=[]
                for tgt_coord_,tgt_label_ in zip(tgt_coord,tgt_label):
                    p=random.random()
                    if p < 0.1:
                        pse_coord.append(tgt_coord_)
                        pse_label.append(tgt_label_)
                    elif p < 0.2:
                        pse_coord.append([0,0,0,0,0,0,0,0])
                        pse_label.append(random.randint(0, 9))
                    else:
                        pse_coord.append([0,0,0,0,0,0,0,0])
                        pse_label.append(0) 
            else:           
                for tgt_coord_ in  tgt_coord:
                        p=random.random()
                        if p < 0.1:
                            pse_coord.append(tgt_coord_) 
                        elif p < 0.2:
                            pse_coord.append([0,0,0,0,0,0,0,0])
                        else:
                            pse_coord.append([0,0,0,0,0,0,0,0])       
            tgt_coord.append([1000,1000,1000,1000,1000,1000,1000,1000])   
            pse_coord.append([1000,1000,1000,1000,1000,1000,1000,1000])  
            if with_label:        
                tgt_label.append(6)
                pse_label.append(6)       
            """ padding """   
            if 513> len(src_coord):             #add pads
                    for j in range(( 513 - len(src_coord))):
                        src_coord.append([0,0,0,0,0,0,0,0])
                        if with_label:
                            src_label.append(0)
            if 511> len(tgt_coord):             #add pads
                    for j in range(( 511 - len(tgt_coord))):
                        tgt_coord.append([0,0,0,0,0,0,0,0])
                        pse_coord.append([0,0,0,0,0,0,0,0])
                        tgt_index.extend([0])
                        if with_label:
                            tgt_label.append(0)
                            pse_label.append(0)
                  
            line_coord_data= src_coord+tgt_coord+pse_coord
            if with_label:
                label_data=src_label+tgt_label+pse_label        
                return line_coord_data,sentence_num+1,tgt_index,label_data
            else:
                return line_coord_data,sentence_num+1,tgt_index
            
        for layout in tqdm.tqdm(examples):
            if with_label:
                input_sentence_coord_,sentence_num_,tgt_index_,label_data_=get_input_data(layout)
                input_label_data=torch.tensor(label_data_,dtype=torch.long)
            else:
                input_sentence_coord_,sentence_num_,tgt_index_=get_input_data(layout)
            input_sentence_coord=torch.tensor(input_sentence_coord_,dtype=torch.long)
            """ judge """
            # judge_coord=input_sentence_coord[:, 5] - input_sentence_coord[ :,1]
            # judge_coord_=input_sentence_coord[:, 4] - input_sentence_coord[:, 0]
            # if torch.lt(judge_coord_, 0).any()==True or torch.lt(judge_coord, 0).any():
            #     continue
            tgt_indexs=torch.tensor(tgt_index_,dtype=torch.long)
            sentence_num=torch.tensor(sentence_num_,dtype=torch.long)
            if sentence_num>510:
                continue
            if with_label:
                feature = {
                    "input_sentence_coord": input_sentence_coord,
                    "target_index": tgt_indexs,
                    "sentences_num":sentence_num,
                    "label_data":input_label_data,
                }
            else:
                feature = {
                    "input_sentence_coord": input_sentence_coord,
                    "target_index": tgt_indexs,
                    "sentences_num":sentence_num,
                }
                
            features.append(feature)
        if shuffle:
            random.shuffle(features)
        if not os.path.exists(os.path.dirname(cached_features_file)):
            os.makedirs(os.path.dirname(cached_features_file))
        print("Saving features into cached file ", cached_features_file)
        torch.save(features, cached_features_file)
    return features
  