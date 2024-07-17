from operator import truediv
import os
from re import T
import sys
import json
from telnetlib import DO
from xmlrpc.client import Boolean
import torch
import json
import tqdm
import random
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import math
os.chdir(sys.path[0]) 
from datetime import datetime 
sys.path.append('..')
from s2s_ft import utils
from s2s_ft.modeling_test import TextLineForS2SDecoder, BertConfig
from torch.utils.data import DataLoader,SequentialSampler

DocType='en'
FileType='test'
BATCH_SIZE=100
REBUILD_SIZE=1024
LINESIZE=0.3
WITH_LABEL=True
BOX=True
#MODEL_PATH=f'./output/output_model/{DocType}_12_768_{WITH_LABEL}/'
MODEL_PATH=f'./output/output_model/m6docbox_12_768_True_m6doc_en/'
TEST_LAYOUT_FILE=f'./datasets/{DocType}/{FileType}/{DocType}_layout_{FileType}.json'
CACHE_FILE=f"datasets/cache/features_{DocType}_{FileType}_{WITH_LABEL}.pt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def LoadCacheLayoutLabel(with_label=True,shuffle=False):
    features = [] 
    examples = []
    def GetInputData(input_line):
        line_coord_data=[]
        sentence_num=[]
        tgt_index=[]
        input_label_data=[]
        src_text = input_line.get("src", "")
        line_coord_data.append(src_text) 
        """ coord """
        for i in range(len(line_coord_data)):
            line_coord_data[i].insert(0,[0,0,0,0])
            line_coord_data[i].append([1000,1000,1000,1000])
            sentence_num.append(len(line_coord_data[i]))
            line_coord_data[i].append([0,0,0,0])
            if 513> len(line_coord_data[i]):             #add pads
                    for j in range(( 513 - len(line_coord_data[i]))):
                        line_coord_data[i].append([0,0,0,0])
        line_coord_data=torch.tensor(line_coord_data).to(dtype=torch.long).to(device)
        input_ids=line_coord_data
        input_ids.to(device)
        """ tgt_index """
        tgt_index_ = input_line.get("tgt_index", "")
        if 511> len(tgt_index_):                
            for j in range(( 511 - len(tgt_index_))):
                tgt_index_.extend([0])
        tgt_index.append(tgt_index_)
        tgt_index=torch.tensor(tgt_index).to(dtype=torch.long).to(device)
        """ sentence num """
        sentence_num=torch.tensor(sentence_num).to(dtype=torch.long).to(device)
        if with_label:
            input_label_data_=input_line.get("src_label_note", "")
            input_label_data.append(input_label_data_)
            """ label """
            for i in range(len(input_label_data)):
                input_label_data[i].insert(0,0)
                input_label_data[i].append(6)
                if 513> len(input_label_data[i]):
                    for j in range(513-len(input_label_data[i])):
                        input_label_data[i].append(0)
            input_label_data_=torch.tensor(input_label_data).to(dtype=torch.long).to(device)  
        else:
            input_label_data_=torch.zeros(513).to(dtype=torch.long).to(device)  
        return input_ids,sentence_num,tgt_index,input_label_data_
    
    if CACHE_FILE is not None and os.path.exists(CACHE_FILE):
        print("Loading features from cached file ", CACHE_FILE)
        features = torch.load(CACHE_FILE) 
    else: 
        print("Creating features from dataset at ", TEST_LAYOUT_FILE)
        time_start=datetime.now()      
        with open(TEST_LAYOUT_FILE, 'r') as file: 
            for i,layout_line in enumerate(file):
                        if (i + 1) % 10000 == 0:
                            print(f'{i + 1} lines ...')
                        examples.append((json.loads(layout_line)))
        
        for input in tqdm.tqdm(examples):
            input_ids,sentences_num,tgt_index,input_label_data=GetInputData(input)
            """ judge """
            judge_coord=input_ids[:, :,-1] - input_ids[ :,:,-3]
            judge_coord_=input_ids[:, :,2] - input_ids[:,:, 0]
            if torch.lt(judge_coord_, 0).any()==True or torch.lt(judge_coord, 0).any():
                continue
            sentence_num=torch.tensor(tgt_index,dtype=torch.long)
            if torch.gt(sentence_num, 508).any()==True:
                continue
            feature = {
                        "input_ids": input_ids,
                        "sentences_num": sentences_num,
                        "tgt_index":tgt_index,
                        "input_label_data":input_label_data,
                    }
            features.append(feature)
        time_end=datetime.now()
        print(f"handle data spend {time_end-time_start}")
        if shuffle:
            random.shuffle(features)
        if not os.path.exists(os.path.dirname(CACHE_FILE)):
            os.makedirs(os.path.dirname(CACHE_FILE))
        print("Saving features into cached file ", CACHE_FILE)
        torch.save(features, CACHE_FILE)
    return features

def LoadCacheBox(with_label=True,shuffle=False):
    features = [] 
    examples = []
    def GetInputData(input_line):
        line_coord_data=[]
        sentence_num=[]
        tgt_index=[]
        input_label_data=[]
        src_text = input_line.get("src", "")
        line_coord_data.append(src_text) 
        """ coord """
        for i in range(len(line_coord_data)):
            line_coord_data[i].insert(0,[0,0,0,0,0,0,0,0])
            line_coord_data[i].append([1000,1000,1000,1000,1000,1000,1000,1000])
            sentence_num.append(len(line_coord_data[i]))
            line_coord_data[i].append([0,0,0,0,0,0,0,0])
            if 513> len(line_coord_data[i]):             #add pads
                    for j in range(( 513 - len(line_coord_data[i]))):
                        line_coord_data[i].append([0,0,0,0,0,0,0,0])
        line_coord_data=torch.tensor(line_coord_data).to(dtype=torch.long).to(device)
        input_ids=line_coord_data
        input_ids.to(device)
        """ tgt_index """
        tgt_index_ = input_line.get("tgt_index", "")
        if 511> len(tgt_index_):                
            for j in range(( 511 - len(tgt_index_))):
                tgt_index_.extend([0])
        tgt_index.append(tgt_index_)
        tgt_index=torch.tensor(tgt_index).to(dtype=torch.long).to(device)
        """ sentence num """
        sentence_num=torch.tensor(sentence_num).to(dtype=torch.long).to(device)
        if with_label:
            input_label_data_=input_line.get("src_label_note", "")
            input_label_data.append(input_label_data_)
            """ label """
            for i in range(len(input_label_data)):
                input_label_data[i].insert(0,0)
                input_label_data[i].append(6)
                if 513> len(input_label_data[i]):
                    for j in range(513-len(input_label_data[i])):
                        input_label_data[i].append(0)
            input_label_data_=torch.tensor(input_label_data).to(dtype=torch.long).to(device)  
        else:
            input_label_data_=torch.zeros(513).to(dtype=torch.long).to(device)  
        return input_ids,sentence_num,tgt_index,input_label_data_
    
    if CACHE_FILE is not None and os.path.exists(CACHE_FILE):
        print("Loading features from cached file ", CACHE_FILE)
        features = torch.load(CACHE_FILE) 
    else: 
        print("Creating features from dataset at ", TEST_LAYOUT_FILE)
        time_start=datetime.now()      
        with open(TEST_LAYOUT_FILE, 'r') as file: 
            for i,layout_line in enumerate(file):
                        if (i + 1) % 10000 == 0:
                            print(f'{i + 1} lines ...')
                        examples.append((json.loads(layout_line)))
        
        for input in tqdm.tqdm(examples):
            input_ids,sentences_num,tgt_index,input_label_data=GetInputData(input)
            """ judge """
            # judge_coord=input_ids[:, :,-1] - input_ids[ :,:,-3]
            # judge_coord_=input_ids[:, :,2] - input_ids[:,:, 0]
            # if torch.lt(judge_coord_, 0).any()==True or torch.lt(judge_coord, 0).any():
            #     continue
            sentence_num=torch.tensor(tgt_index,dtype=torch.long)
            if torch.gt(sentence_num, 508).any()==True:
                continue
            feature = {
                        "input_ids": input_ids,
                        "sentences_num": sentences_num,
                        "tgt_index":tgt_index,
                        "input_label_data":input_label_data,
                    }
            features.append(feature)
        time_end=datetime.now()
        print(f"handle data spend {time_end-time_start}")
        if shuffle:
            random.shuffle(features)
        if not os.path.exists(os.path.dirname(CACHE_FILE)):
            os.makedirs(os.path.dirname(CACHE_FILE))
        print("Saving features into cached file ", CACHE_FILE)
        torch.save(features, CACHE_FILE)
    return features

def RebuildImage(origin_coord,origin_dic,predict_dic,each_accuarcy,index):
    _, ax = plt.subplots()
    ax.xaxis.set_ticks_position('top')  
    ax.invert_yaxis()  
    ax.set_xlim(0, REBUILD_SIZE)
    ax.set_ylim(REBUILD_SIZE, 0)
    ax.set_aspect('equal', adjustable='box')
    ax.axis('off')
    """ draw origin rectangle """
    for coord in origin_coord:
        x1, y1, x2, y2 = coord
        rect = plt.Rectangle((x1,y1),x2-x1,y2-y1 ,linewidth=LINESIZE, edgecolor='black', facecolor='none')
        ax.add_patch(rect)
    """draw arrows black mean predict corrcet red mean predict wrong blue mean the origin"""            
    for key in origin_dic:
        start_point=key
        if key in predict_dic and predict_dic[key]==origin_dic[key]:
            vector=predict_dic[key]
            end_point = (start_point[0] + vector[0], start_point[1] + vector[1])
            plt.arrow(start_point[0], start_point[1], end_point[0] - start_point[0], end_point[1] - start_point[1],
                head_width=5, head_length=5, linewidth=0.5, color='black')
        else:
            vector_ori=origin_dic[key]
            end_point_ori = (start_point[0] + vector_ori[0], start_point[1] + vector_ori[1])
            ax.annotate("", xy=(end_point_ori[0], end_point_ori[1]), xytext=(start_point[0], start_point[1]),
                        arrowprops=dict(arrowstyle="->,head_length=0.05,head_width=0.05", linewidth=LINESIZE,color="blue"))
            if key in predict_dic:
                vector_pre=predict_dic[key]
                end_point_pre = (start_point[0] + vector_pre[0], start_point[1] + vector_pre[1])
                ax.annotate("", xy=(end_point_pre[0], end_point_pre[1]), xytext=(start_point[0], start_point[1]),
                        arrowprops=dict(arrowstyle="->,head_length=0.05,head_width=0.05", linewidth=LINESIZE,color="red"))            
    text = ("origin_boxs_num : " + str(len(origin_dic)) +
            "\npredict num :" + str(len(predict_dic)) +
            "\nAccuarcy: \n" + str(each_accuarcy[index]) + '%')
    plt.text(REBUILD_SIZE - 250, 100, text, color="black",fontsize=3)
    plt.savefig(f'./output/model_test_output_images/{index}_tgt_coord.png', pad_inches=0,dpi=800)

def RebuildImageBox(origin_coord,origin_dic,predict_dic,each_accuarcy,index):
    _, ax = plt.subplots()
    ax.xaxis.set_ticks_position('top')  
    ax.invert_yaxis()  
    ax.set_xlim(0, REBUILD_SIZE)
    ax.set_ylim(REBUILD_SIZE, 0)
    ax.set_aspect('equal', adjustable='box')
    ax.axis('off')
    """ draw origin rectangle """
    for coord in origin_coord:
        x1, y1, x3,y3,x2, y2,x4,y4 = coord
        x = [x1, x3, x2, x4]
        y = [y1, y3, y2, y4]
        rect = Polygon(xy=list(zip(x, y)), closed=True, linewidth=LINESIZE, edgecolor='black', facecolor='none')
        ax.add_patch(rect)
    """draw arrows black mean predict corrcet red mean predict wrong blue mean the origin"""            
    for key in origin_dic:
        start_point=key
        if key in predict_dic and predict_dic[key]==origin_dic[key]:
            vector=predict_dic[key]
            end_point = (start_point[0] + vector[0], start_point[1] + vector[1])
            plt.arrow(start_point[0], start_point[1], end_point[0] - start_point[0], end_point[1] - start_point[1],
                head_width=5, head_length=5, linewidth=0.5, color='black')
        else:
            vector_ori=origin_dic[key]
            end_point_ori = (start_point[0] + vector_ori[0], start_point[1] + vector_ori[1])
            ax.annotate("", xy=(end_point_ori[0], end_point_ori[1]), xytext=(start_point[0], start_point[1]),
                        arrowprops=dict(arrowstyle="->,head_length=0.05,head_width=0.05", linewidth=LINESIZE,color="blue"))
            if key in predict_dic:
                vector_pre=predict_dic[key]
                end_point_pre = (start_point[0] + vector_pre[0], start_point[1] + vector_pre[1])
                ax.annotate("", xy=(end_point_pre[0], end_point_pre[1]), xytext=(start_point[0], start_point[1]),
                        arrowprops=dict(arrowstyle="->,head_length=0.05,head_width=0.05", linewidth=LINESIZE,color="red"))            
    text = ("origin_boxs_num : " + str(len(origin_dic)) +
            "\npredict num :" + str(len(predict_dic)) +
            "\nAccuarcy: \n" + str(each_accuarcy[index]) + '%')
    plt.text(REBUILD_SIZE - 250, 100, text, color="black",fontsize=3)
    plt.savefig(f'./output/model_test_output_images/{index}_tgt_coord.png', pad_inches=0,dpi=800)
    
def CalScale(seq):
    max_elements = max(max(sublist) for sublist in seq)
    scale=max_elements/(REBUILD_SIZE-1)
    return scale

def ARD(seq1, seq2):
    if len(seq1) != len(seq2):
        raise ValueError("Sequences must have the same length")
    seq_ori=list(range(len(seq1)))
    seq_predict=[]
    seq_dict={}
    for index,seq_ in enumerate(seq1):
        seq_dict[seq_]=index
    for index,seq_ in enumerate(seq2):
        if seq_ in seq_dict:
            seq_predict.append(seq_dict[seq_])
        else:
            seq_predict.append(-1)
    ard_sum=0
    for seq_ori_,seq_predict_ in zip(seq_ori, seq_predict):
        if seq_predict_==-1:
            ard_sum=ard_sum+len(seq1)
        else:
            ard_sum=ard_sum+abs(seq_ori_-seq_predict_)
    return ard_sum/len(seq1)

def PrintModelInfo(model):
    """Print the parameter size and shape of model detail"""
    total_params = 0
    for name, param in model.named_parameters():
        num_params = torch.prod(torch.tensor(param.shape)).item() * param.element_size() / (1024 * 1024)  # 转换为MB
        print(f"{name}: {num_params:.4f} MB, Shape: {param.shape}")
        total_params += num_params
    print(f"Total number of parameters: {total_params:.4f} MB")
    
def main():
    accuarcy_sum=0
    each_accuarcy_=[] 
    """load model and config"""
    config = BertConfig.from_json_file(MODEL_PATH+'config.json',layoutlm_only_layout_flag=True)
    model = TextLineForS2SDecoder(config=config).to(device)
    model.load_state_dict(torch.load(MODEL_PATH+'pytorch_model.bin'), strict=False)
    #PrintModelInfo(model)    
    """create dataloader"""
    if BOX:
        feature=LoadCacheBox(with_label=WITH_LABEL,shuffle=True)
    else:
        feature=LoadCacheLayoutLabel(with_label=WITH_LABEL,shuffle=True)
    dataset=utils.TextLineDataset(features=feature,with_label=WITH_LABEL)
    sampler = SequentialSampler(dataset) 
    test_dataloader = DataLoader(dataset, sampler=sampler,batch_size=BATCH_SIZE)
    """ Test! """
    print("  ************************ Running Test ***********************")
    print("  Num Samples = ", len(feature))
    print("  Batch size per node = ", BATCH_SIZE)
    print(f"  model info:{config.num_hidden_layers} layers,{config.hidden_size} hidden size")
    print(f"  Model is {MODEL_PATH}")
    print(f"  Data is {TEST_LAYOUT_FILE}")
    print("  ****************************************************************")
    train_iterator = tqdm.tqdm(test_dataloader, initial=0,desc="progress", disable=False)
    ard_score_sum=0
    for step,batch in enumerate(train_iterator):
        """get input """
        input_ids,sentences_num,tgt_index,input_label_data= tuple(t.to(device) for t in batch)
        input_ids=input_ids.squeeze(1)
        sentences_num=sentences_num.squeeze(1)
        input_label_data=input_label_data.squeeze(1)  
        tgt_index=tgt_index.squeeze(1)   
        inputs = {  'input_ids':input_ids,
                    'sentence_num': sentences_num,
                    'input_label_data':input_label_data,
                    'WITH_LABEL':WITH_LABEL}
        """model test"""
        model.eval()
        with torch.no_grad():
            traces = model(**inputs)
            each_accuarcy=[] 
            for index_ in range(len(traces)):
                """ard"""
                origin_order=tgt_index[index_][:sentences_num[index_]-2]
                predict_order=traces[index_][:sentences_num[index_]-2]
                ard_score=ARD(origin_order.tolist(),predict_order.tolist())
                ard_score_sum=ard_score_sum+ard_score
                """"""
                origin_dic={}
                predict_dic={}
                ori_tensor = torch.index_select(input_ids[index_,1:,0:], dim=0, index=tgt_index[index_])
                reordered_tensor = torch.index_select(input_ids[index_,1:,0:], dim=0, index=traces[index_]) 
                origin_coord = ori_tensor[:sentences_num[index_]-2,:].tolist()
                predict_coord = reordered_tensor[:sentences_num[index_]-2,:].tolist()
                """scale"""
                scale=CalScale(origin_coord)
                predict_coord = [[int(element//scale) for element in sublist] for sublist in predict_coord]
                origin_coord = [[int(element//scale) for element in sublist] for sublist in origin_coord]
                """create centerpoint_vector  dict """
                vector_value_sum=0
                accuarcy_vector_sum=0
                if BOX:
                    """box"""
                    origin_center_coord=[((x2+x1+x3+x4)/4,(y2+y1+y3+y4)/4) for x1,y1,x2,y2,x3,y3,x4,y4 in origin_coord]
                    predict_center_coord=[((x2+x1+x3+x4)/4,(y2+y1+y3+y4)/4) for x1,y1,x2,y2,x3,y3,x4,y4 in predict_coord]
                else:
                    """layout"""
                    origin_center_coord=[((x2+x1)/2,(y2+y1)/2) for x1,y1,x2,y2 in origin_coord]
                    predict_center_coord=[((x2+x1)/2,(y2+y1)/2) for x1,y1,x2,y2 in predict_coord]
                for i in range(1,len(predict_center_coord)):
                    if i==len(predict_center_coord):
                        if origin_center_coord[i] not in origin_dic:
                            origin_dic[origin_center_coord[i-1]]=(0,0)
                        if predict_center_coord[i] not in predict_dic:
                            predict_dic[predict_center_coord[i-1]]=(0, 0)
                    else:
                        odx = origin_center_coord[i][0] - origin_center_coord[i-1][0]
                        ody = origin_center_coord[i][1] - origin_center_coord[i-1][1]
                        pdx = predict_center_coord[i][0] - predict_center_coord[i-1][0]
                        pdy = predict_center_coord[i][1] - predict_center_coord[i-1][1]
                        if origin_center_coord[i-1] not in origin_dic:
                            origin_dic[origin_center_coord[i-1]]=(odx,ody)
                            vector_value_sum+=math.sqrt(odx**2+ody**2)
                        if predict_center_coord[i-1] not in predict_dic:
                            predict_dic[predict_center_coord[i-1]]=(pdx, pdy)
                """calculate accuarcy"""
                for key in origin_dic:
                    if key in predict_dic and predict_dic[key]==origin_dic[key]:
                        accuarcy_vector_sum+=(math.sqrt(predict_dic[key][0]**2+predict_dic[key][1]**2)/vector_value_sum)     
                accuarcy_sum+=accuarcy_vector_sum*100
                #print(f'{step*BATCH_SIZE+index_+1} acc is {accuarcy_vector_sum*100}')
                each_accuarcy.append(accuarcy_vector_sum*100)
                each_accuarcy_.append(accuarcy_vector_sum*100) 
                """rebuild image """
                if BOX:
                    RebuildImageBox(origin_coord,origin_dic,predict_dic,each_accuarcy,index_)
                # else:
                #     #RebuildImage(origin_coord,origin_dic,predict_dic,each_accuarcy,index_)
                #     print()
        ard_score_averge=ard_score_sum/len(each_accuarcy_)
        accuarcy_averge=accuarcy_sum/len(each_accuarcy_)
        train_iterator.set_description('AverAcc：%9.7f %% Ard:%9.7f ' % (accuarcy_averge,ard_score_averge))                           
    print(f"all {dataset.num_instances} samples avearge accuarcy is {accuarcy_sum/dataset.num_instances} !")    
     
if __name__ =="__main__":
    main()