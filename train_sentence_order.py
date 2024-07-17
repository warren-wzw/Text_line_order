import os
import sys
from xmlrpc.client import Boolean 
import torch
import tqdm
import re
from datetime import datetime
os.chdir(sys.path[0])

sys.path.append('..')
from s2s_ft import utils
from s2s_ft.modeling_train import TextLineOrderS2S
from s2s_ft.config import BertForSeq2SeqConfig
from torch.utils.data import (DataLoader, SequentialSampler)
from transformers import AdamW, get_linear_schedule_with_warmup
from s2s_ft.modeling_test import BertConfig
try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter

LR=7e-5
"""dataset"""
DOC_TYPE='m6docbox'
FILE_TYPE="train"
EPOCH_NUM=30
BATCH_SIZE=15
WITH_LABEL=True
BOX=True
if DOC_TYPE=='inclined':
    BOX=True
"""model"""
LAYERNUM=6
HIDDENSIZE=192
TensorBoardStep=50
#PRETRAINED_MODEL_PATH=f"./output/output_model/{DOC_TYPE}_{LAYERNUM}_{HIDDENSIZE}_{WITH_LABEL}/"
PRETRAINED_MODEL_PATH=f"./output/output_model/m6docbox_6_192_True_m6doc_en/"
SAVEMODEL=f"./output/output_model/{DOC_TYPE}_{LAYERNUM}_{HIDDENSIZE}_{WITH_LABEL}_m6doc_en/"
OUTPUT_MODEL_DIR='./output/output_model'
TRAIN_FILE=f'./datasets/{DOC_TYPE}/train'
VALITATION_FILE=f'./datasets/{DOC_TYPE}/val'
cached_features_readingbank_train=f'./datasets/cache/Features_{DOC_TYPE}_{FILE_TYPE}_{WITH_LABEL}.pt'
cached_features_readingbank_val=f'./datasets/cache/Features_{DOC_TYPE}_val_{WITH_LABEL}.pt'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def PrintModelInfo(model):
    """Print the parameter size and shape of model detail"""
    total_params = 0
    for name, param in model.named_parameters():
        num_params = torch.prod(torch.tensor(param.shape)).item() * param.element_size() / (1024 * 1024)  # 转换为MB
        print(f"{name}: {num_params:.4f} MB, Shape: {param.shape}")
        total_params += num_params
    print(f"Total number of parameters: {total_params:.4f} MB")

def get_model():
    model_config = BertConfig.from_json_file(PRETRAINED_MODEL_PATH+'config.json',layoutlm_only_layout_flag=True)
    config = BertForSeq2SeqConfig.from_exist_config(
        config=model_config, label_smoothing=0.1,
        max_position_embeddings=1024,max_source_length=513,  
        base_model_type='layoutlm',layoutlm_only_layout_flag=False,
    )
    """change config"""
    config.hidden_size=HIDDENSIZE
    config.num_hidden_layers=LAYERNUM
    """load model """
    model = TextLineOrderS2S(config=config).to(device)
    model.load_state_dict(torch.load(PRETRAINED_MODEL_PATH+"pytorch_model.bin"), strict=False)
    #PrintModelInfo(model)
    return model,config 

def CreateDataloader(instance_file,cached_file):
    if BOX:
        features = utils.load_and_cache_box_label(example_path=instance_file,
                 with_label=WITH_LABEL,shuffle=True,cached_features_file=cached_file)
    else:
        features = utils.load_and_cache_only_layout(example_path=instance_file,
                    with_label=WITH_LABEL,shuffle=True,cached_features_file=cached_file)    
    dataset = utils.Seq2seqDatasetForLayoutlm_self(features=features,
                num_training_instances=len(features),with_label=WITH_LABEL)
    sampler = SequentialSampler(dataset) 
    data_loader = DataLoader(dataset, sampler=sampler,batch_size=BATCH_SIZE,
                collate_fn=utils.batch_list_to_batch_tensors_self)
    return data_loader

def main():
    """ get model & config """
    model,config=get_model()
    """ create dataloader """
    train_dataloader = CreateDataloader(TRAIN_FILE,cached_features_readingbank_train)
    validation_dataloader = CreateDataloader(VALITATION_FILE,cached_features_readingbank_val)
    """ create optimizer """
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
    optimizer = AdamW(optimizer_grouped_parameters, lr=LR, eps=1e-8)
    Total_training_step=(train_dataloader.sampler.data_source.num_training_instances//BATCH_SIZE)*EPOCH_NUM
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=Total_training_step//30,num_training_steps=Total_training_step, last_epoch=-1)
    """ Train! """
    print("  ************************ Running training ***********************")
    print("  Num Epochs = ", EPOCH_NUM)
    print("  Batch size per node = ", BATCH_SIZE)
    print("  Num examples = ", train_dataloader.sampler.data_source.num_training_instances)
    print(f"  model info:{config.num_hidden_layers} layers,{config.hidden_size} hidden size")
    print(f"  Pretrained Model is {PRETRAINED_MODEL_PATH}")
    print(f"  Save Model as {SAVEMODEL}")
    print("  ****************************************************************")
    model.train()
    model.zero_grad()
    torch.cuda.empty_cache()    
    tb_writer = SummaryWriter(log_dir='./output/tf-logs/') 
    for epoch_index in range(EPOCH_NUM):
        global_step=0
        sum_loss=0.0
        sum_accuarcy=0.0
        train_iterator = tqdm.tqdm(train_dataloader, initial=0,desc="Iter (loss=X.XXX, lr=X.XXXXXXX)", disable=False)
        for step, batch in enumerate(train_iterator): 
            if WITH_LABEL:   
                input_sentence_coord,target_index,sentence_num,input_label_data= tuple(t.to(device) for t in batch)        
                inputs = {  'input_sentence_coord':input_sentence_coord,
                            'sentence_num': sentence_num,
                            'target_index': target_index,
                            'input_label_data':input_label_data}
            else:
                input_sentence_coord,target_index,sentence_num= tuple(t.to(device) for t in batch)
                inputs = {  'input_sentence_coord':input_sentence_coord,
                            'sentence_num': sentence_num,
                            'target_index': target_index,
                            'input_label_data':None}
            loss,Accuarcy=model(**inputs)
            global_step += 1  
            sum_loss += loss.item()
            sum_accuarcy+=Accuarcy*100
            current_lr= scheduler.get_last_lr()[0]
            train_iterator.set_description('Epoch=%d,loss=%5.3f Acc= %9.7f %% lr=%9.7f' % (epoch_index, loss.item(),Accuarcy*100,current_lr))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0) 
            optimizer.step()
            scheduler.step()
            model.zero_grad()      
            """ tensorbooard """
            if  global_step % TensorBoardStep== 0 and tb_writer is not None:
                tb_writer.add_scalar('train/lr', current_lr, global_step=global_step)
                tb_writer.add_scalar('train/loss', loss.item(), global_step=global_step)
        print("Average loss is ",
              sum_loss/(train_dataloader.sampler.data_source.num_training_instances//BATCH_SIZE+1e-5),
              "Accuarcy is ",
              sum_accuarcy/(train_dataloader.sampler.data_source.num_training_instances//BATCH_SIZE+1e-5))
        """ validation """
        best_validation_accuarcy=0   
        total_validation_loss = 0.0
        total_validation_accuarcy = 0.0
        model.eval()
        with torch.no_grad():
            validation_iterator = tqdm.tqdm(validation_dataloader, initial=0,desc="Iter (loss=X.XXX, lr=X.XXXXXXX)", disable=False)
            for step, batch in enumerate(validation_iterator):
                if WITH_LABEL: 
                    input_sentence_coord, target_index, sentence_num,input_label_data = tuple(t.to(device) for t in batch)          
                    inputs = {  'input_sentence_coord':input_sentence_coord,
                                'sentence_num': sentence_num,
                                'target_index': target_index,
                                'input_label_data':input_label_data}
                else:
                    input_sentence_coord, target_index, sentence_num = tuple(t.to(device) for t in batch)
                    inputs = {  'input_sentence_coord':input_sentence_coord,
                                'sentence_num': sentence_num,
                                'target_index': target_index,
                                'input_label_data':None}
                loss, Accuarcy = model(**inputs)
                total_validation_loss += loss.item()
                total_validation_accuarcy += Accuarcy
        average_validation_accuarcy = total_validation_accuarcy / len(validation_dataloader)
        print("Validation accuarcy is ",average_validation_accuarcy*100," %")
        if average_validation_accuarcy >best_validation_accuarcy:
            best_validation_accuarcy = average_validation_accuarcy
            best_mdoel = os.path.join(SAVEMODEL)
            os.makedirs(best_mdoel, exist_ok=True)
            model.save_pretrained(best_mdoel)
            print("Saving model checkpoint {} into {} at {}".format(global_step, best_mdoel, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))   

if __name__ =="__main__":
    main()