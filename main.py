# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 14:20:36 2023

@author: user
"""

import pandas as pd
import numpy as np

from Preprocess import make_binary_mixtures
from Model import load_classification_model, load_regression_model, load_MDR_model, make_prediction, predict_module


encoding = 'utf-8-sig'

# input mixture 파일
input_path = #input path
model_path = #model path
output_path = #output path
output_filename = '플랫폼용예시데이터_결과.xlsx'

# reader로 읽어서 dataframe 저장
df = pd.read_excel(input_path)

# make_binary_mixtures로 2종조합 생성 후 아래 경로에 저장
# 상대적 비율과 절대적 비율 두 버전 저장
binary_mixtures = make_binary_mixtures(df, output_path, 'rel')

endpoints = ['ER', 'AR', 'THR', 'NPC', 'EB']
for endpoint in endpoints:
    binary_mixtures = predict_module(binary_mixtures, model_path, output_path, endpoint)

#binary_mixtures.to_csv(output_path + output_filename, index=False, encoding=encoding)
binary_mixtures.to_excel(output_path + output_filename, index=False)

