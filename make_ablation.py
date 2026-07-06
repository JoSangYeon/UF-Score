import pandas as pd
from pydub import AudioSegment
import json
from tqdm import tqdm
import os
import numpy as np
from concurrent.futures import ThreadPoolExecutor

valor = pd.read_csv('Valor_32k_train.csv')
vggsound = pd.read_csv('VGG_200k_train.csv')

cs_cols = ['img_lan_CS', 'aud_lan_CS', 'img_aud_CS']

valor['CS_mean'] = valor[cs_cols].mean(axis=1)
valor['CS_var'] = valor[cs_cols].var(axis=1) 

vggsound['CS_mean'] = vggsound[cs_cols].mean(axis=1)
vggsound['CS_var'] = vggsound[cs_cols].var(axis=1)

alpha_values = [0.1, 0.9]
alpha_values = [0.5]

# valor_root_path = '/mnt/storage/ufscore/workspace/_DATASET/VALOR-32K'
# for alpha in alpha_values:
#     valor['CS_score'] = valor['CS_mean'] - alpha * valor['CS_var']
#     sorted_valor = valor.sort_values(by='CS_score', ascending=False)
#     ratio = [0.5, 0.75, 0.9]
#     for r in ratio:
#         data = []
#         valor_filtered = sorted_valor.head(int(len(sorted_valor) * r))
#         for row_index, img_path, wav_path, caption, _, _, _, _, _, _, _, _, _, in tqdm(valor_filtered.values, desc=f"Alpha: {alpha}, Ratio: {r}"):
#             wav = AudioSegment.from_file(os.path.join(valor_root_path, wav_path))
#             temp = {
#             'key': str(row_index).zfill(6),
#             'img': os.path.join(valor_root_path, img_path),
#             'wav': os.path.join(valor_root_path, wav_path),
#             'label': caption,
#             'wav_length' : len(wav) / 1000.0,
#             'text_length' : len(caption.split())
#             }
#             data.append(temp)
#         print(f"Ratio: {r}, len(data): {len(data)}")
        
#         with open(f'/mnt/storage/ufscore/workspace/UF_Score/data/ablation_data/valor/va_train_{alpha}_{r}.data', 'w') as f:
#             for entry in data:
#                 json.dump(entry, f)
#                 f.write('\n')
                
                
# vgg_root_path = '/mnt/storage/ufscore/workspace/_DATASET/vgg_sound'
# for alpha in alpha_values:
#     vggsound['CS_score'] = vggsound['CS_mean'] - alpha * vggsound['CS_var']
#     vggsound_sorted = vggsound.sort_values(by='CS_score', ascending=False)
#     ratio = [0.5, 0.75, 0.9]
#     for r in ratio:
#         data = []
#         vggsound_filtered = vggsound_sorted.head(int(len(vggsound_sorted) * r))
        
#         # 병렬 처리를 위한 데이터 미리 준비
#         process_data = []
#         for row in tqdm(vggsound_filtered.values):
#             ID, _, img_path, wav_path, _, _, caption, *_ = row
#             process_data.append((ID, img_path, wav_path, caption))
            
#         # 오디오 파일 일괄 처리
#         from concurrent.futures import ThreadPoolExecutor
        
#         def process_row(args):
#             ID, img_path, wav_path, caption = args
#             wav = AudioSegment.from_file(os.path.join(vgg_root_path, wav_path))
#             return {
#                 'key': str(ID).zfill(6),
#                 'img': os.path.join(vgg_root_path, img_path), 
#                 'wav': os.path.join(vgg_root_path, wav_path),
#                 'label': caption,
#                 'wav_length': len(wav) / 1000.0,
#                 'text_length': len(caption.split())
#             }
            
#         with ThreadPoolExecutor(max_workers=8) as executor:
#             data = list(executor.map(process_row, process_data))
            
#         # 결과 저장
#         output_path = f'/mnt/storage/ufscore/workspace/UF_Score/data/ablation_data/vggsound/va_train_{int(abs(alpha)*10)}_{int(r*100)}.data'
#         with open(output_path, 'w') as f:
#             for entry in data:
#                 json.dump(entry, f)
#                 f.write('\n')        


root_path = os.path.join('..', '_DATASET', 'VALOR-32K')
train_path = 'Valor_32k_train_blip2_score.csv'
data = pd.read_csv(train_path)

# fdf = pd.read_csv(train_path)
# cs_mean_list = []
# cs_var_list = []
# g_len_mean_list = []
# g_len_var_list = []

# for _, _, _, _, _, _, _, ilcs, alcs, iacs, igl, agl, tgl in fdf.values:
#     cs_mean_list.append(np.mean([ilcs, alcs, iacs]))
#     cs_var_list.append(np.var([ilcs, alcs, iacs]))
#     g_len_mean_list.append(np.mean([igl, agl, tgl]))
#     g_len_var_list.append(np.var([igl, agl, tgl]))
    
# fdf['CS_mean'] = cs_mean_list
# fdf['CS_var'] = cs_var_list
# fdf['CS_score'] = fdf['CS_mean'] - 0.5 * fdf['CS_var']
# fdf['g_len_mean'] = g_len_mean_list
# fdf['g_len_var'] = g_len_var_list
# fdf['g_len_score'] = fdf['g_len_mean'] - 0.5 * fdf['g_len_var']

fdf_sorted = data.sort_values(by=["BLIP2_score"], ascending=[False])

fdf_list = []
split_rate = [0.9, 0.75, 50]
for rate in split_rate:
    length = len(fdf_sorted)
    sep_idx = int(length * rate)
    
    tempt_data = fdf_sorted.iloc[:sep_idx, :]
    import ipdb; ipdb.set_trace()
    def process_row(row):
        st,img_path,wav_path,caption,new_caption,*_ = row
        wav = AudioSegment.from_file(os.path.join(root_path, wav_path))
        cap = caption if pd.isna(new_caption) else new_caption
        return {
            'key': str(st).zfill(4),
            'img': os.path.join(root_path, img_path),
            'wav': os.path.join(root_path, wav_path), 
            'label': caption,
            'wav_length': len(wav) / 1000.0,
            'text_length': len(cap.split())
        }

    # 멀티프로세싱으로 데이터 처리 속도 향상
    with ThreadPoolExecutor(max_workers=8) as executor:
        data = list(tqdm(executor.map(process_row, tempt_data.values), total=len(tempt_data)))
        
    output_path = f'/mnt/storage/ufscore/workspace/UF_Score/data/blip_data/va_train_{int(rate*100)}.data'
    with open(output_path, 'w') as f:
        for entry in data:
            json.dump(entry, f)
            f.write('\n')