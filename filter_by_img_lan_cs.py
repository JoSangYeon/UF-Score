import os
import sys
import json
import random
import concurrent.futures

import numpy as np
import pandas as pd

from tqdm import tqdm

# pydub 대신 librosa 사용하거나 기본값 사용
try:
    import librosa
    USE_LIBROSA = True
except ImportError:
    USE_LIBROSA = False
    print("librosa not found, using default wav_length = 10.0")

def set_SEED(SEED):
    random.seed(SEED)
    os.environ['PYTHONHASHSEED'] = str(SEED)
    np.random.seed(SEED)

set_SEED(SEED=17)

def _get_statistics(data):
    wav_length_list = []
    text_legnth_list = []
    for temp in data:
        wav_length_list.append(temp['wav_length'])
        text_legnth_list.append(temp['text_length'])
    
    return np.round(np.mean(wav_length_list), 2), np.round(np.mean(text_legnth_list), 2)

def get_wav_length(wav_path):
    """
    오디오 파일의 길이를 구하는 함수
    """
    if USE_LIBROSA:
        try:
            y, sr = librosa.load(wav_path)
            return len(y) / sr
        except Exception as e:
            print(f"Error loading {wav_path} with librosa: {e}")
            return 10.0  # 기본값
    else:
        # librosa가 없으면 기본값 10.0 사용
        return 10.0

def make_data_file(df, root_path):
    """
    데이터프레임을 .data 형식으로 변환하는 함수 (va_test.data와 동일한 형식)
    """
    data = []
    #for (ID, startseconds, img_path, wav_path, label, raw_caption, caption, img_lan_CS, aud_lan_CS, img_aud_CS, img_g_len, aud_g_len, txt_g_len) in tqdm(df.values):
    for (_, row_index,img_path,wav_path,caption,img_lan_CS,aud_lan_CS,img_aud_CS,img_g_len,aud_g_len,txt_g_len, _) in tqdm(df.values):
        try:
            full_wav_path = os.path.join(root_path, wav_path)
            wav_length = get_wav_length(full_wav_path)
            
            temp = {
                'key': str(ID).zfill(6),
                #'key': str(row_index).zfill(6),
                'img': os.path.join(root_path, img_path),
                'wav': full_wav_path,
                'label': caption,
                'wav_length': wav_length,
                'text_length': len(caption.split())
            }
            data.append(temp)
        except Exception as e:
            print(f"Error processing {wav_path}: {e}")
            continue
    
    return data

def save_data(data, name, save_path=os.path.join('.', 'data', 'clip_score_img')):    
    """
    데이터를 .data 파일로 저장하는 함수 (va_test.data와 동일한 형식)
    """
    os.makedirs(save_path, exist_ok=True)
    file_path = os.path.join(save_path, name)

    with open(file_path, 'w') as file:
        for entry in data:
            json.dump(entry, file)
            file.write('\n')

    avg_wav_len, avg_txt_len = _get_statistics(data)
    print(f"{file_path} >>> avg. wav length : {avg_wav_len} | avg. txt length : {avg_txt_len}")
    
    return data

def filter_by_img_lan_cs(csv_path, root_path, top_percentages=[90, 75, 50], num_processes=5):
    """
    img_lan_CS 점수를 기반으로 상위 N%의 데이터를 필터링하여 .data 형식으로 저장
    
    Args:
        csv_path: CSV 파일 경로
        root_path: 데이터 루트 경로
        top_percentages: 상위 N% 리스트 (예: [90, 75, 50, 25])
        num_processes: 멀티프로세싱 프로세스 수
    """
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} samples")
    
    # img_lan_CS 점수를 기준으로 내림차순 정렬
    df_sorted = df.sort_values(by=["img_lan_CS"], ascending=[False])
    print(f"Data sorted by img_lan_CS score")
    print(f"img_lan_CS score range: {df_sorted['img_lan_CS'].min():.4f} ~ {df_sorted['img_lan_CS'].max():.4f}")
    
    # 각 퍼센트별로 데이터 필터링 및 저장
    for percentage in top_percentages:
        print(f"\n=== Processing top {percentage}% data ===")
        
        # 상위 N% 데이터 선택
        top_n_count = int(len(df_sorted) * (percentage / 100))
        top_n_df = df_sorted.iloc[:top_n_count, :]
        
        print(f"Selected {top_n_count} samples (top {percentage}%)")
        print(f"img_lan_CS threshold: {top_n_df['img_lan_CS'].min():.4f}")
        
        # 멀티프로세싱을 위해 데이터를 분할
        try:
            df_list = np.array_split(top_n_df, num_processes)
        except:
            df_list = [top_n_df]  # 분할할 수 없으면 전체 데이터 사용
        
        # 멀티프로세싱으로 데이터 처리
        procs = []
        data_list = []
        
        if len(df_list) > 1:
            pool = concurrent.futures.ProcessPoolExecutor(max_workers=len(df_list))
            for idx, df_chunk in enumerate(df_list):
                procs.append(pool.submit(make_data_file, df_chunk, root_path, idx))
            
            for p in concurrent.futures.as_completed(procs):
                data_list.append(p.result())
            
            pool.shutdown()
        else:
            # 단일 프로세스
            data_list.append(make_data_file(df_list[0], root_path, 0))
        
        # 모든 데이터 병합 및 셔플
        data = sum(data_list, [])
        random.shuffle(data)
        
        # .data 파일로 저장
        filename = f'va_train_img_lan_cs_{percentage:03d}.data'
        save_data(data, name=filename, save_path=os.path.join('.', 'data', 'clip_score_ic'))
        
        print(f"Saved {len(data)} samples to {filename}")


def filter_by_aud_lan_cs(csv_path, root_path, top_percentages=[90, 75, 50], num_processes=5):
    """
    aud_lan_CS 점수를 기반으로 상위 N%의 데이터를 필터링하여 .data 형식으로 저장
    
    Args:
        csv_path: CSV 파일 경로
        root_path: 데이터 루트 경로
        top_percentages: 상위 N% 리스트 (예: [90, 75, 50, 25])
        num_processes: 멀티프로세싱 프로세스 수
    """
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} samples")
    
    # img_lan_CS 점수를 기준으로 내림차순 정렬
    df_sorted = df.sort_values(by=["aud_lan_CS"], ascending=[False])
    print(f"Data sorted by aud_lan_CS score")
    print(f"aud_lan_CS score range: {df_sorted['aud_lan_CS'].min():.4f} ~ {df_sorted['aud_lan_CS'].max():.4f}")
    
    # 각 퍼센트별로 데이터 필터링 및 저장
    for percentage in top_percentages:
        print(f"\n=== Processing top {percentage}% data ===")
        
        # 상위 N% 데이터 선택
        top_n_count = int(len(df_sorted) * (percentage / 100))
        top_n_df = df_sorted.iloc[:top_n_count, :]
        
        print(f"Selected {top_n_count} samples (top {percentage}%)")
        print(f"aud_lan_CS threshold: {top_n_df['aud_lan_CS'].min():.4f}")
        
        # 멀티프로세싱을 위해 데이터를 분할
        try:
            df_list = np.array_split(top_n_df, num_processes)
        except:
            df_list = [top_n_df]  # 분할할 수 없으면 전체 데이터 사용
        
        # 멀티프로세싱으로 데이터 처리
        procs = []
        data_list = []
        
        if len(df_list) > 1:
            pool = concurrent.futures.ProcessPoolExecutor(max_workers=len(df_list))
            for idx, df_chunk in enumerate(df_list):
                procs.append(pool.submit(make_data_file, df_chunk, root_path, idx))
            
            for p in concurrent.futures.as_completed(procs):
                data_list.append(p.result())
            
            pool.shutdown()
        else:
            # 단일 프로세스
            data_list.append(make_data_file(df_list[0], root_path, 0))
        
        # 모든 데이터 병합 및 셔플
        data = sum(data_list, [])
        random.shuffle(data)
        
        # .data 파일로 저장
        filename = f'vs_train_aud_lan_cs_{percentage:03d}.data'
        save_data(data, name=filename, save_path=os.path.join('.', 'data', 'clip_score_ac'))
        
        print(f"Saved {len(data)} samples to {filename}")

def main():
    # 설정
    csv_file = 'VGG_200k_train.csv'  # CSV 파일명
    root_path = os.path.join('..', '_DATASET', 'vgg_sound')  # 데이터 루트 경로
    #csv_file = 'Valor_32k_train.csv'  # CSV 파일명
    #root_path = os.path.join('..', '_DATASET', 'VALOR-32K')  # 데이터 루트 경로
    
    # 상위 N% 설정 (90%, 75%, 50%, 25%)
    #top_percentages = [90, 75, 50]
    top_percentages = [80]
    
    # CSV 파일 존재 확인
    if not os.path.exists(csv_file):
        print(f"Error: {csv_file} not found!")
        return
    
    # 데이터 루트 경로 존재 확인
    if not os.path.exists(root_path):
        print(f"Error: {root_path} not found!")
        return
    
    print("=== img_lan_CS 기반 데이터 필터링 시작 ===")
    print(f"CSV 파일: {csv_file}")
    print(f"데이터 루트: {root_path}")
    print(f"필터링 퍼센트: {top_percentages}%")
    #print(f"저장 경로: ./data/clip_score/")
    
    # 필터링 실행
    filter_by_aud_lan_cs(
        csv_path=csv_file,
        root_path=root_path,
        top_percentages=top_percentages,
        num_processes=5
    )
    
    print("\n=== 모든 처리 완료 ===")

if __name__ == "__main__":
    main() 