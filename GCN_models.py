# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 13:51:20 2023

@author: user
"""

import numpy as np

#Dataset
from rdkit import Chem
from rdkit.Chem.rdmolops import GetMolFrags
from torch.utils.data import Dataset
from rdkit.Chem.rdmolops import GetAdjacencyMatrix

#Model
import torch
import torch.nn as nn
import torch.nn.functional as F



######## GCN dataset ########
# Molecular Graph 생성
class GCNDataset(Dataset):
    
    def __init__(self, max_num_atoms, smiles_list_A, smiles_list_B, ratio_A, ratio_B, mf_A, mf_B):
        self.max_num_atoms = max_num_atoms
        self.smiles_list_A = smiles_list_A
        self.smiles_list_B = smiles_list_B
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
        return len(self.mf_A)
    
    def __getitem__(self, idx):
        sample = dict()
        sample['x1'] = self.input_feature_list_A[idx]
        sample['x2'] = self.input_feature_list_B[idx]
        sample['r1'] = self.ratio_A[idx]
        sample['r2'] = self.ratio_B[idx]
        sample['adj1'] = self.adj_list_A[idx]
        sample['adj2'] = self.adj_list_B[idx]
        sample['mf1'] = self.mf_A[idx]
        sample['mf2'] = self.mf_B[idx]
        return sample


######## GCN model ########

#GCN model blocks
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
    
    
#Classification model structure
class GCNNet_Classification(torch.nn.Module):
    def __init__(self, n_atom, n_conv,n_MLP, n_mf, n_atom_feature, n_conv_feature, n_readout_feature, n_feature, n_feature2, use_bn, use_mf, use_dropout, drop_rate, concat):
        super(GCNNet_Classification, self).__init__()
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
        #x = torch.cat([r1[:,None] * x1, r2[:,None] * x2], dim=1) # concat
            
        for layer in self.MLP_list:
            x = layer(x)
        
        retval = self.fc(x)
        retval = self.sigmoid(retval)
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

#Regression MDR prediction model structure
class GCNNet_MDR(torch.nn.Module):
    def __init__(self, n_atom, n_conv,n_MLP, n_mf, n_atom_feature, n_conv_feature, n_readout_feature, n_feature, n_feature2, use_bn, use_mf, use_dropout, drop_rate, concat):
        super(GCNNet_MDR, self).__init__()
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
            
            
        if self.concat == 'aug':
            x = torch.cat([r1[:,None] * x1, r2[:,None] * x2], dim=1) # concat하고 뒤집어서 한번더 학습
        elif not self.concat:
            x = r1[:,None] *x1 + r2[:,None] *x2 # sum
        else: #concat 큰거 앞으로 붙이기
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