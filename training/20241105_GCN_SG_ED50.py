# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 13:58:41 2023

@author: user
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold

from rdkit import Chem
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem.rdmolops import GetMolFrags
from rdkit.Chem import Descriptors
#from rdkit.Chem.Crippen import MolLogP


#Dataset
from torch.utils.data import Dataset, DataLoader, random_split
from rdkit.Chem.rdmolops import GetAdjacencyMatrix

import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import random

#mordred
from datetime import datetime

######## GCN dataset ########

class GCNDataset(Dataset):
    
    def __init__(self, max_num_atoms, smiles_list_A, smiles_list_B, ratio_A, ratio_B, mf_A, mf_B, y_list):
        self.max_num_atoms = max_num_atoms
        self.use_mf = use_mf
        self.smiles_list_A = smiles_list_A
        self.smiles_list_B = smiles_list_B
        self.y_list = torch.from_numpy(np.array(y_list))
        self.input_feature_list_A = []
        self.input_feature_list_B = []
        self.ratio_A = ratio_A
        self.ratio_B = ratio_B
        self.adj_list_A = []
        self.adj_list_B = []
        self.mf_A = mf_A
        self.mf_B = mf_B
        self.process_data()
        self.mf_A = list(map(torch.from_numpy, np.array(mf_A, dtype = np.float64)))
        self.mf_B = list(map(torch.from_numpy, np.array(mf_B, dtype = np.float64)))
        
    def process_data(self):
        self.mol_to_graph(self.smiles_list_A, self.input_feature_list_A, self.adj_list_A)
        self.mol_to_graph(self.smiles_list_B, self.input_feature_list_B, self.adj_list_B)

    def mol_to_graph(self, smi_list, feature_list, adj_list):
        max_num_atoms = self.max_num_atoms
        for smiles in smi_list:
            mol = Chem.MolFromSmiles(smiles)
            mol = self.remove_salt(mol)
            num_atoms = mol.GetNumAtoms()
            #Get padded adj
            #max atom수만큼 0000000을 padding
            adj = GetAdjacencyMatrix(mol) + np.eye(num_atoms)
            
            #degree 높은애들은 계속 더해지니까 normalize 해줌 (DADWH)
            Degree_tilde = 1/np.sqrt(adj.sum(1) + 1) * np.eye(num_atoms)
            norm_adj = Degree_tilde @ adj @ Degree_tilde
            
            padded_adj = np.zeros((max_num_atoms, max_num_atoms))
            padded_adj[:num_atoms, :num_atoms] = norm_adj
            
            #Get property list
            feature = []
            for i in range(num_atoms):
                feature.append(self.get_atom_feature(mol, i))
            feature = np.array(feature)
            
            padded_feature = np.zeros((max_num_atoms,feature.shape[1]))
            padded_feature[:num_atoms,:feature.shape[1]] = feature
            
            feature_list.append(torch.from_numpy(padded_feature))
            adj_list.append(torch.from_numpy(padded_adj))

    def onehot_encoding(self, x, allowable_set):
        #Maps inputs not inthe allowable set to the last element
        if x not in allowable_set:
            x = allowable_set[-1]
        return list(map(lambda s: x==s, allowable_set))
    
    def get_atom_feature(self, m, atom_i):
        atom = m.GetAtomWithIdx(atom_i)
        symbol = self.onehot_encoding(atom.GetSymbol(),['C','N','O','F','Cl','Br','I','S','P','Na','ELSE']) # 10
        chirality = self.onehot_encoding(str(atom.GetChiralTag()), ['CHI_UNSPECIFIED', 'CHI_TETRAHEDRAL_CW', 'CHI_TETRAHEDRAL_CCW'])
        hy = self.onehot_encoding(atom.GetTotalNumHs(), [0,1,2,3])
        degree = self.onehot_encoding(atom.GetDegree(), [0,1,2,3,4])
        num2 = self.onehot_encoding(atom.GetImplicitValence(), [0,1,2,3])
        num = self.onehot_encoding(atom.GetTotalValence(),[0,1,2,3,4,5,6])
        hybrid = self.onehot_encoding(str(atom.GetHybridization()), ['SP','SP2','SP3','ELSE'])
        etc = [atom.IsInRing(), atom.GetIsAromatic(), atom.GetFormalCharge()]
        return np.array(symbol + chirality+ hy + degree + num2 + num + hybrid + etc)
    
    def remove_salt(self, mol):
        mols = list(GetMolFrags(mol, asMols=True))
        if mols:
            mols.sort(reverse = True, key = lambda m:m.GetNumAtoms())
            mol = mols[0]
        return mol

    
    def __len__(self):
        return len(self.y_list)
    
    def __getitem__(self, idx):
        sample = dict()
        sample['x1'] = self.input_feature_list_A[idx]
        sample['x2'] = self.input_feature_list_B[idx]
        sample['r1'] = self.ratio_A[idx]
        sample['r2'] = self.ratio_B[idx]
        sample['adj1'] = self.adj_list_A[idx]
        sample['adj2'] = self.adj_list_B[idx]
        sample['y'] = self.y_list[idx]
        sample['mf1'] = self.mf_A[idx]
        sample['mf2'] = self.mf_B[idx]
        return sample


######## GCN model ########

class GCNLayer(torch.nn.Module):
    def __init__(self, n_atom, in_dim, out_dim, use_bn):
        super(GCNLayer, self).__init__()
        self.n_atom = n_atom
        self.in_dim = in_dim
        self.out_dim = out_dim        
        self.use_bn = use_bn
        self.linear = nn.Linear(self.in_dim, self.out_dim)
        self.bn = nn.BatchNorm1d(self.n_atom)
        
    def forward(self, x, adj):
        x = self.linear(x)
        x = torch.matmul(adj, x)
        if self.use_bn:
            x = self.bn(x)
        retval = F.relu(x)
        return retval

class Readout(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Readout, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.linear = nn.Linear(self.in_dim, self.out_dim)
        
    def forward(self, x):
        x = self.linear(x)
        x = x.sum(1)
        retval = F.relu(x)
        return retval

class Predictor(torch.nn.Module):
    def __init__(self, in_dim, out_dim, use_dropout, drop_rate):
        super(Predictor, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.use_dropout = use_dropout
        self.drop_rate = drop_rate
        self.linear = nn.Linear(self.in_dim, self.out_dim)
        self.dropout = nn.Dropout(self.drop_rate)
        
    def forward(self, x):
        x = self.linear(x)
        retval = F.relu(x)
        if self.use_dropout:
            retval = self.dropout(retval)
        return retval
    
#Regression PC10 prediction model structure
class GCNNet_Regression(torch.nn.Module):
    def __init__(self, n_atom, n_conv, n_MLP, n_mf, n_atom_feature, n_conv_feature, n_readout_feature, n_feature, n_feature2, use_bn, use_mf, use_dropout, drop_rate, concat):
        super(GCNNet_Regression, self).__init__()
        self.n_atom = n_atom
        self.n_atom_feature = n_atom_feature
        self.n_conv = n_conv
        self.n_MLP = n_MLP
        self.n_conv_feature = n_conv_feature
        self.embedding = nn.Linear(self.n_atom_feature, self.n_conv_feature)
        self.use_bn = use_bn
        self.concat = concat
        
        GCN_list = []
        for i in range(self.n_conv):
            GCN_list.append(GCNLayer(self.n_atom, self.n_conv_feature, self.n_conv_feature, self.use_bn))
        self.GCN_list = nn.ModuleList(GCN_list)
        self.n_readout_feature = n_readout_feature
        self.readout = Readout(self.n_conv_feature, self.n_readout_feature)
        self.use_mf= use_mf
        self.n_mf = n_mf
        self.n_feature = n_feature
        self.n_feature2 = n_feature2
        self.drop_rate = drop_rate
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(self.drop_rate)
        if self.use_mf:
            self.in_feature = self.n_readout_feature + self.n_mf
        else:
            self.in_feature = self.n_readout_feature
        if concat:
            self.in_feature = self.in_feature * 2
        MLP_list = []
        for i in range(self.n_MLP):
            if i==0:
                MLP_list.append(Predictor(self.in_feature, self.n_feature, self.use_dropout, self.drop_rate))
            else:
                MLP_list.append(Predictor(self.n_feature, self.n_feature, self.use_dropout, self.drop_rate))
        self.MLP_list = nn.ModuleList(MLP_list)
        self.fc = nn.Linear(self.n_feature2, 1)
        self.sigmoid = nn.Sigmoid()
        
        
    def forward(self, x1, x2, r1, r2, adj1, adj2, mf1, mf2):
        x1 = self.embedding(x1)
        x2 = self.embedding(x2)
        for layer in self.GCN_list:
            x1 = layer(x1, adj1)
            x2 = layer(x2, adj2)
        x1 = self.readout(x1)
        x2 = self.readout(x2)
        if self.use_mf:
            x1 = torch.cat([x1, mf1], dim=1)
            x2 = torch.cat([x2, mf2], dim=1)

        
        if not self.concat:
            x = r1[:,None] *x1 + r2[:,None] *x2 # sum
        else: #concat
            x = torch.zeros(torch.cat([x1,x2],dim=1).shape).cuda()
            
            for i in range(len(r1)):
                if r1[i]>=0.5:
                    x[i,:int(x.shape[1]/2)] = x1[i] * r1[i]
                    x[i,int(x.shape[1]/2):] = x2[i] * r2[i]
                else:
                    x[i,:int(x.shape[1]/2)] = x2[i] * r2[i]
                    x[i,int(x.shape[1]/2):] = x1[i] * r1[i]
    
        for layer in self.MLP_list:
            x = layer(x)
        
        retval = self.fc(x)
        return retval
######## Data Loading ########



#load data
dt = pd.read_excel('C:\\Users\\user\\Desktop\\1\\Modeling\\21. SG 혼합물 독성 예측\\input\\SG_binary_EC50_data_with_SMILES.xlsx')
mf_A = pd.read_csv('C:\\Users\\user\\Desktop\\1\\Modeling\\21. SG 혼합물 독성 예측\\input\\SG_EC50_mf_A.csv')
mf_B = pd.read_csv('C:\\Users\\user\\Desktop\\1\\Modeling\\21. SG 혼합물 독성 예측\\input\\SG_EC50_mf_B.csv')
print(dt.shape)

smi_A = list(dt['smi_A'])
smi_B = list(dt['smi_B'])
y_list = -np.log10(list(dt['ED50.result']))
ratio_A = list(dt['Comp_Sub_A'])
ratio_B = list(dt['Comp_Sub_B'])
temp = []
for i, smi in enumerate(set(smi_A + smi_B)):
    m = Chem.MolFromSmiles(smi)
    num_a = len(m.GetAtoms())
    temp.append(num_a)
max_natoms = max(temp)
max_natoms = 256

#train/test split
num_data = len(y_list)
num_train = int(num_data * 0.8)
num_val = int(num_data * 0.1)
num_test = num_data - num_train - num_val


#hyperparameter
seeds = [1,2,10,20,42,420,777,1022,10004,99999]
use_bn, use_mf, use_dropout, drop_rate, concat = True, False, True, 0.3, False
seed, batch_size, num_epoch, lr = 1, 32, 300, 5e-4

#for train
mfmf = [False, True]
drdr = [[False, 0], [True, 0.2]]
lrlr = [[300,1e-3], [500,5e-4], [1000,5e-4]]
mlps = [1,2,3]
max_natoms = 256

# for Test
if True:
    drdr = [[False,0]]
    seeds=[99999]
    mfmf = [True]
    lrlr = [[500,5e-4]]
    mlps =[3]

for n_MLP in mlps:    
    for use_mf in mfmf:
        for num_epoch, lr in lrlr:
            for use_dropout, drop_rate in drdr:
                total_results = []
                for seed in seeds:
                    print(f'seed {seed} start!')
                    dataset = GCNDataset(max_natoms, smi_A, smi_B, ratio_A, ratio_B, mf_A, mf_B, y_list)
                    
                    n_conv = 3 
                    n_mf, n_atom_feature, n_conv_feature, n_readout_feature, n_feature, n_feature2 = dataset.mf_A[0].shape[0], dataset.input_feature_list_A[0].shape[1], 64, 64, 64, 64
                    
                    #seed
                    seed = seed
                    random.seed(seed)
                    np.random.seed(seed)
                    torch.manual_seed(seed)
                    torch.cuda.manual_seed_all(seed)
                    
                    
                    
                    
                    
                    train_dataset, val_dataset, test_dataset = random_split(dataset, [num_train, num_val, num_test])
                    print(f"Training Data Size : {len(train_dataset)}")
                    print(f"Validation Data Size : {len(val_dataset)}")
                    print(f"Testing Data Size : {len(test_dataset)}")
                    
                    idx_train = train_dataset.indices
                    idx_val = val_dataset.indices
                    idx_test = test_dataset.indices
                    
                    scaler_A = MinMaxScaler()
                    scaler_B = MinMaxScaler()
                    
                    dt_mf_A = np.array(dataset.mf_A)
                    dt_mf_B = np.array(dataset.mf_B)
                    
                    dt_mf_A[idx_train] = scaler_A.fit_transform(dt_mf_A[idx_train])
                    dt_mf_B[idx_train] = scaler_B.fit_transform(dt_mf_B[idx_train])
                    dt_mf_A[idx_val] = scaler_A.transform(dt_mf_A[idx_val])
                    dt_mf_B[idx_val] = scaler_B.transform(dt_mf_B[idx_val])
                    dt_mf_A[idx_test] = scaler_A.transform(dt_mf_A[idx_test])
                    dt_mf_B[idx_test] = scaler_B.transform(dt_mf_B[idx_test])
                    
                    train_dataset.dataset.mf_A = dt_mf_A
                    train_dataset.dataset.mf_B = dt_mf_B
                
                    

                    
                    #Dataloader
                    train_dataloader = DataLoader(train_dataset, batch_size = batch_size)
                    val_dataloader = DataLoader(val_dataset, batch_size = batch_size)
                    test_dataloader = DataLoader(test_dataset, batch_size = batch_size)
                    
                    

                    #Make model
                    model = GCNNet_Regression(max_natoms, 
                                n_conv = n_conv, 
                                n_mf = n_mf,  n_MLP = n_MLP,
                                n_atom_feature = n_atom_feature,
                                n_conv_feature = n_conv_feature, 
                                n_readout_feature = n_readout_feature, 
                                n_feature = n_feature, 
                                n_feature2 = n_feature2, 
                                use_bn = use_bn, use_mf = use_mf, use_dropout = use_dropout, drop_rate = drop_rate, concat=concat)
                    
                    
                    
                    #Training parameter 
                    lr = lr
                    weight_decay = 0
                    optimizer = torch.optim.Adam(model.parameters(), lr=lr)#, weight_decay= weight_decay)
                    num_epoch = num_epoch
                    loss_fn = nn.MSELoss()
                    
                    
                    loss_list = []
                    st = time.time()
                    
                    model.cuda()
                    train_loss = []
                    val_loss = []
                    for epoch in range(num_epoch):
                        model.train()
                        loss_list = []
                        train_targets = []
                        train_outputs = []
                        for i_batch, batch in enumerate(train_dataloader):
                            x1 = batch['x1'].cuda().float()
                            x2 = batch['x2'].cuda().float()
                            r1 = batch['r1'].cuda().float()
                            r2 = batch['r2'].cuda().float()
                            y = batch['y'].cuda().float()
                            adj1 = batch['adj1'].cuda().float()
                            adj2 = batch['adj2'].cuda().float()
                            mf1 = batch['mf1'].cuda().float()
                            mf2 = batch['mf2'].cuda().float()
                            pred = model(x1, x2, r1, r2, adj1, adj2, mf1, mf2).squeeze(-1)
                            loss = loss_fn(pred,y)
                            optimizer.zero_grad()
                            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                            loss.backward()
                            optimizer.step()
                            loss_list.append(loss.data.cpu().numpy())
                            train_targets.append(y.cpu())
                            train_outputs.append(pred.cpu().detach())
                        i_loss = np.mean(np.array(loss_list))
                        train_loss.append(i_loss)
                        #print(epoch + 1 , i_loss)
                        train_targets = np.concatenate(train_targets)
                        train_outputs = np.concatenate(train_outputs)
                        
                        model.eval()
                        with torch.no_grad():
                            val_loss_list = []
                            for i_batch, batch in enumerate(val_dataloader):
                                x1 = batch['x1'].cuda().float()
                                x2 = batch['x2'].cuda().float()
                                r1 = batch['r1'].cuda().float()
                                r2 = batch['r2'].cuda().float()
                                y = batch['y'].cuda().float()
                                adj1 = batch['adj1'].cuda().float()
                                adj2 = batch['adj2'].cuda().float()
                                mf1 = batch['mf1'].cuda().float()
                                mf2 = batch['mf2'].cuda().float()
                                pred = model(x1, x2, r1, r2, adj1, adj2, mf1, mf2).squeeze(-1)
                                loss = loss_fn(pred,y)
                                val_loss_list.append(loss.data.cpu().numpy())
                            if (epoch+1) % 1 == 0 :
                                print(epoch + 1, i_loss,np.mean(np.array(val_loss_list)))
                            val_loss.append(np.mean(np.array(val_loss_list)))
            
                    # myplot = plt.plot(np.linspace(1, num_epoch,num_epoch).astype(int),train_loss,np.linspace(1, num_epoch,num_epoch).astype(int),val_loss)
                    # plt.show()
            
                    end = time.time()
                    print ('Time:',end-st)
                    model.eval()
                    with torch.no_grad():
                        test_loss_list = []
                        test_targets = []
                        test_outputs = []
                        for i_batch, batch in enumerate(test_dataloader):
                            x1 = batch['x1'].cuda().float()
                            x2 = batch['x2'].cuda().float()
                            r1 = batch['r1'].cuda().float()
                            r2 = batch['r2'].cuda().float()
                            y = batch['y'].cuda().float()
                            adj1 = batch['adj1'].cuda().float()
                            adj2 = batch['adj2'].cuda().float()
                            mf1 = batch['mf1'].cuda().float()
                            mf2 = batch['mf2'].cuda().float()
                            pred = model(x1, x2, r1, r2, adj1, adj2, mf1, mf2).squeeze(-1)
                            loss = loss_fn(pred,y)
                            test_loss_list.append(loss.data.cpu().numpy())
                            test_targets.append(y.cpu())
                            test_outputs.append(pred.cpu().detach())
                        test_mse = np.mean(np.array(test_loss_list))
                        print('test loss : ', test_mse)
                        test_targets = np.concatenate(test_targets)
                        test_outputs = np.concatenate(test_outputs)
            
                    test_rmse = np.sqrt(test_mse)
            
                    def draw_and_r2(pred, real):
                        
                        r2 = (1 - sum((pred- real)**2) / sum((real - np.mean(np.array(real)))**2)).item()
                        print('rsqaured : ', r2)
                        
                        return r2

                    
                    r2_unlog_train = draw_and_r2(10**(-1*train_outputs),10**(-1*train_targets)) 
                    r2_train = draw_and_r2(train_outputs, train_targets)
                    r2_unlog_test = draw_and_r2(10**(-1*test_outputs),10**(-1*test_targets)) 
                    r2_test = draw_and_r2(test_outputs, test_targets)
                    now = datetime.now().strftime('%Y/%m/%d %H:%M:%S')
                    

                    i_results = [seed, now, n_MLP, batch_size, num_epoch, lr, 
                                n_conv, n_mf, n_atom_feature, n_conv_feature, 
                                n_readout_feature, n_feature, n_feature2, 
                                use_bn, use_mf, use_dropout, drop_rate, weight_decay, concat, 
                                test_mse, test_rmse,r2_train, r2_unlog_train,r2_test, r2_unlog_test]
                    
                    
            
                    results = pd.read_excel('C:\\Users\\user\\Desktop\\1\\Modeling\\21. SG 혼합물 독성 예측\\output\\20241105_SG_ED50_results.xlsx')
                    results.loc[results.shape[0]] = i_results
                    results.to_excel('C:\\Users\\user\\Desktop\\1\\Modeling\\21. SG 혼합물 독성 예측\\output\\20241105_SG_ED50_results.xlsx',index=False)
            
                    total_results.append(i_results)
            
                total_results = np.array(total_results)
                summary = ['Average']
                summary += (i_results[1:19])
                summary += (list(np.mean(np.array(total_results)[:,19:].astype(float), axis=0)))
                
                results = pd.read_excel('C:\\Users\\user\\Desktop\\1\\Modeling\\21. SG 혼합물 독성 예측\\output\\20241105_SG_ED50_results.xlsx')
                results.loc[results.shape[0]] = summary
                results.to_excel('C:\\Users\\user\\Desktop\\1\\Modeling\\21. SG 혼합물 독성 예측\\output\\20241105_SG_ED50_results.xlsx',index=False)
                
                print(summary)


import pickle
import os 
os.chdir(r'C:\Users\user\Desktop\1\Modeling\20. 혼합독성 예측 모음\SG\ED50')
if True:
    torch.save(model.state_dict(), 'SG_ED50_weights.pth')
    pickle.dump(scaler_A,open('scaler_A.sav', 'wb'))
    pickle.dump(scaler_B,open('scaler_B.sav', 'wb'))
       



