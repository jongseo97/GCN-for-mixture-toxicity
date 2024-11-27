# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 13:22:32 2024

@author: user
"""

import torch
import numpy as np

from GCN_models import GCNNet_Classification, GCNNet_Regression, GCNNet_MDR
from Preprocess import preprocessing





def load_classification_model(model_path, endpoint):
    
    # 모델 생성
    if endpoint == 'ER':
        #n_atom, n_conv,n_MLP, n_mf, n_atom_feature, n_conv_feature, n_readout_feature, n_feature, n_feature2, use_bn, use_mf, use_dropout, drop_rate, concat
        model = GCNNet_Classification(256,3,2,195,41,64,64,64,64,False,False,True,0.3,False)
    if endpoint == 'AR':
        model = GCNNet_Classification(256,3,1,195,41,64,64,64,64,False,False,True,0.3,False)
    if endpoint == 'THR':
        model = GCNNet_Classification(256,3,1,195,41,64,64,64,64,False,True,False,0,False)
    if endpoint == 'NPC':
        model = GCNNet_Classification(256,3,1,195,41,64,64,64,64,False,False,False,0,False)
    if endpoint == 'EB':
        model = GCNNet_Classification(256,3,1,195,41,64,64,64,64,False,False,False,0,False)
    if endpoint == 'SG':
        model = GCNNet_Classification(256,3,1,195,41,64,64,64,64,True,False,True,0.2,False)
    # 모델 로딩
    model_path = model_path + '/' + endpoint + '/Classification/' + endpoint + '_Classification_weights.pth'
    
    # CPU/GPU 확인
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    #GPU 사용
    if torch.cuda.is_available():
        model.cuda()
    #파라미터 고정
    model.eval()
    
    return model



def load_regression_model(model_path, endpoint):
    
    # 모델 생성
    if endpoint == 'ER':
        #n_atom, n_conv,n_MLP, n_mf, n_atom_feature, n_conv_feature, n_readout_feature, n_feature, n_feature2, use_bn, use_mf, use_dropout, drop_rate, concat
        model = GCNNet_Regression(256,3,1,195,41,64,64,64,64,False,True,False,0,False)
    if endpoint == 'AR':
        model = GCNNet_Regression(256,3,1,195,41,64,64,64,64,False,True,False,0,False)
    if endpoint == 'THR':
        model = GCNNet_Regression(256,3,2,195,41,64,64,64,64,False,False,False,0,False)
    if endpoint == 'NPC':
        model = GCNNet_Regression(256,3,1,195,41,64,64,64,64,False,False,False,0,False)
    if endpoint == 'EB':
        model = GCNNet_Regression(256,3,2,195,41,64,64,64,64,False,True,False,0,False)
    if endpoint == 'SG':
        model = GCNNet_Regression(256,3,3,195,41,64,64,64,64,True,True,False,0,False)    

    # 모델 로딩
    model_path = model_path + '/' + endpoint + '/Regression/' + endpoint + '_Regression_weights.pth'
    # CPU/GPU 확인
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    #GPU 사용
    if torch.cuda.is_available():
        model.cuda()
    #파라미터 고정
    model.eval()
    
    return model


def load_MDR_model(model_path, endpoint):
    
    # 모델 생성
    if endpoint == 'ER':
        #n_atom, n_conv,n_MLP, n_mf, n_atom_feature, n_conv_feature, n_readout_feature, n_feature, n_feature2, use_bn, use_mf, use_dropout, drop_rate, concat
        model = GCNNet_MDR(256,3,2,195,41,64,64,64,64,True,False,True,0.3,False) # seed 99999
    if endpoint == 'AR':
        model = GCNNet_MDR(256,3,3,195,41,64,64,64,64,True,False,True,0.2,False) # seed 1022
    if endpoint == 'THR':
        model = GCNNet_MDR(256,3,3,195,41,64,64,64,64,True,False,True,0.2,False) # seed 10004
    if endpoint == 'NPC':
        model = GCNNet_MDR(256,3,1,195,41,64,64,64,64,True,False,False,0,False)
    if endpoint == 'EB':
        model = GCNNet_MDR(256,3,1,195,41,64,64,64,64,True,True,False,0,False)
    if endpoint == 'SG':
        model = GCNNet_MDR(256,3,1,195,41,64,64,64,64,True,True,False,0,False)
    
    # 모델 로딩
    model_path = model_path + '/' + endpoint + '/MDR/' + endpoint + '_MDR_weights.pth'
    # CPU/GPU 확인
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    #GPU 사용
    if torch.cuda.is_available():
        model.cuda()
    #파라미터 고정
    model.eval()
    
    return model




# make prediction results using loaded model
def make_prediction(model, dataloader):
    #모델 구동
    with torch.no_grad():
        for i_batch, batch in enumerate(dataloader):
            # GPU 사용
            if torch.cuda.is_available():
                x1 = batch['x1'].cuda().float()
                x2 = batch['x2'].cuda().float()
                r1 = batch['r1'].cuda().float()
                r2 = batch['r2'].cuda().float()
                adj1 = batch['adj1'].cuda().float()
                adj2 = batch['adj2'].cuda().float()
                mf1 = batch['mf1'].cuda().float()
                mf2 = batch['mf2'].cuda().float()
            # CPU 사용
            else:
                x1 = batch['x1'].float()
                x2 = batch['x2'].float()
                r1 = batch['r1'].float()
                r2 = batch['r2'].float()
                adj1 = batch['adj1'].float()
                adj2 = batch['adj2'].float()
                mf1 = batch['mf1'].float()
                mf2 = batch['mf2'].float()
            
            pred = model(x1, x2, r1, r2, adj1, adj2, mf1, mf2).squeeze(-1)
            
    return pred


def predict_module(binary_mixtures, model_path, output_path, endpoint):
    c_dataloader = preprocessing(binary_mixtures, model_path, endpoint, 'Classification')
    c_model = load_classification_model(model_path, endpoint)
    pred_c = make_prediction(c_model, c_dataloader)
    toxicity = [1 if p>=0.5 else 0 for p in pred_c]
    binary_mixtures[endpoint + '_toxicity_prediction'] = toxicity
    
    mdr_dataloader = preprocessing(binary_mixtures, model_path, endpoint, 'MDR')
    mdr_model = load_MDR_model(model_path, endpoint)
    pred_mdr = make_prediction(mdr_model, mdr_dataloader)
    if endpoint == 'EB':
        pred_mdr = np.exp(1) ** pred_mdr
    mdr = [2 if p>=2 else (1 if p>0.5 else 0) for p in pred_mdr]
    mdr = ['None' if toxicity[i]==0 else mdr[i] for i in range(len(mdr))]
    binary_mixtures[endpoint + '_MDR_class_prediction'] = mdr
    
    r_dataloader = preprocessing(binary_mixtures, model_path, endpoint, 'Regression')
    r_model = load_regression_model(model_path, endpoint)
    pred_r = make_prediction(r_model, r_dataloader)
    pred_r = 10**(-pred_r)
    pred_r = ['None' if toxicity[i]==0 else ('>100,000' if pred_r[i] > 100000 else float(pred_r[i])) for i in range(len(pred_r))]
    if endpoint in ['ER', 'THR']:
        name = endpoint + '_PC10_conc(uM)'
    else:
        name = endpoint + '_IC30_conc(uM)'
    binary_mixtures[name] = pred_r
    
    return binary_mixtures




