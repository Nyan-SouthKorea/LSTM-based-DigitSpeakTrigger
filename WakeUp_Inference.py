from tqdm import tqdm
from natsort import natsorted
import os
from IPython.display import Audio
import librosa
import copy
import random
import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import pyaudio
from collections import deque
import threading
import wave
import torch.nn.functional as F

# cuda 설정
if torch.cuda.is_available() == True:
    device = 'cuda:0'
    print('현재 가상환경 cuda 설정 가능')
else:
    device = 'cpu'
    print('현재 가상환경 cpu 사용')

path = './'
n_mfcc = 40
stream_save = False
model_path = f'{path}/model_lstm-best.pt'
class_list_want = ['backward', 'bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 'follow', 'forward', 'four', 'go', 'happy', 'house', 
                   'learn', 'left', 'marvin', 'nine', 'no', 'off', 'on', 'one', 'right', 'seven', 'sheila', 'six', 'stop', 'three', 'tree', 
                   'two', 'up', 'visual', 'wow', 'yes', 'zero']
class_list_want = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'noise']

# 모델 선언
class ModifiedLSTM(nn.Module):
    def __init__(self, class_list_want):
        super(ModifiedLSTM, self).__init__()
        self.lstm1 = nn.LSTM(input_size=32, hidden_size=128, num_layers=1, batch_first=True)
        self.bn1 = nn.BatchNorm1d(128)
        
        self.lstm2 = nn.LSTM(input_size=128, hidden_size=256, num_layers=1, batch_first=True)
        self.bn2 = nn.BatchNorm1d(256)
        
        self.lstm3 = nn.LSTM(input_size=256, hidden_size=512, num_layers=1, batch_first=True)
        self.bn3 = nn.BatchNorm1d(512)
        self.dropout1 = nn.Dropout(0.3)

        self.lstm4 = nn.LSTM(input_size=512, hidden_size=1024, num_layers=1, batch_first=True)
        self.bn4 = nn.BatchNorm1d(1024)
        
        # 최종 출력을 위한 선형 레이어
        self.fc = nn.Linear(1024, len(class_list_want))  # n개의 

    def forward(self, x):
        x, _ = self.lstm1(x)
        x = self.bn1(x.transpose(1, 2)).transpose(1, 2)
        
        x, _ = self.lstm2(x)
        x = self.bn2(x.transpose(1, 2)).transpose(1, 2)
        
        x, _ = self.lstm3(x)
        x = self.bn3(x.transpose(1, 2)).transpose(1, 2)
        x = self.dropout1(x)

        x, _ = self.lstm4(x)
        x = self.bn4(x.transpose(1, 2)).transpose(1, 2)
        
        # 마지막 시퀀스의 출력만을 사용
        x = self.fc(x[:, -1, :])
        return x
    
# 데이터셋 로드
class AudioDataset(Dataset):
    def __init__(self, mfcc): # 입력으로 변환된 mfcc를 넣는다(batch는 당연히 1)
        self.labels = []
        self.data = []
        self.mfcc = mfcc

    def __len__(self):
        return len(self.mfcc)
    
    def __getitem__(self, idx):
        # np.save(f'./test.npy', self.mfcc)
        # mfccs = np.load(f'./test.npy')
        mfccs = self.normalize_data(self.mfcc)  # MFCC 정규화
        mfccs = torch.tensor(mfccs, dtype=torch.float32)  # PyTorch 텐서로 변환

        return mfccs

    def normalize_data(self, data):
        # Min-Max 정규화
        return (data - data.min()) / (data.max() - data.min())

# 오디오 버퍼 관리
class buffer:
    def __init__(self, stream, chunk, rate, n_mfcc, stream_save):
        self.buffer = deque()
        self.stream = stream
        self.chunk = chunk
        self.rate = rate
        self.n_mfcc = n_mfcc
        self.cnt = 0
        self.stream_save = stream_save
        # 실시간 출력 설정
        self.pyaudio_instance = pyaudio.PyAudio()
        self.output_stream = self.pyaudio_instance.open(format=pyaudio.paInt16,channels=1,rate=self.rate,output=True)


    def stream_start(self):
        print('스트리밍 시작') 
        while True:  
            data = self.stream.read(self.chunk) # 0.1초의 데이터가 빠짐
            self.buffer.append(data) # deque 데이터에 추가
            self.output_stream.write(data)

    def popleft(self):
        buffer_list = list(self.buffer)[:10] # deque를 리스트로 변환 후 처음 10개 요소 슬라이싱
        _ = self.buffer.popleft() # 선입선출 하나 제거
        buffer_list = b''.join(buffer_list) # 버퍼 리스트를 하나의 리스트로 합쳐서 반환
        # 바이트 배열을 np.float32 타입의 오디오 신호로 변환
        audio_signal = np.frombuffer(buffer_list, dtype=np.int16).astype(np.float32) / 32768.0  # 정규화
        # librosa를 사용하여 오디오 길이를 조정하고 MFCC 변환
        # 여기서는 process_audio 함수와 유사한 작업을 메모리에서 직접 수행합니다.
        audio_signal = librosa.util.fix_length(audio_signal, size=self.rate)  # 오디오 길이를 1초로 조정
        # MFCC 변환
        mfccs = librosa.feature.mfcc(y=audio_signal, sr=self.rate, n_mfcc=n_mfcc)
        # 저장 테스트
        if self.stream_save == True:
            if os.path.isdir(f'./test') == False:
                os.makedirs(f'./test', exist_ok = True)
            with wave.open(f'./test/{self.cnt}.wav', 'wb') as wf:
                wf.setnchannels(channels)
                wf.setsampwidth(audio.get_sample_size(format))
                wf.setframerate(self.rate)
                wf.writeframes(buffer_list)
                self.cnt += 1
        return mfccs

# 오디오 길이 맞춰주는 함수
def process_audio(audio_path):
    data, sr = librosa.load(audio_path, sr=None) # 오디오를 원본 샘플링 레이트로 로드
    # 목표 길이 설정 (1초)
    target_length = sr  # 1초에 해당하는 샘플 수
    current_length = len(data) # 현재 데이터 길이
    if current_length > target_length: # 1초를 초과하면 뒷부분을 잘라냄
        data = data[:target_length] 
    elif current_length < target_length: # 1초 미만이면 무음을 추가
        padding = np.zeros(target_length - current_length)
        data = np.concatenate((data, padding))
    return data, sr

# 오디오 설정
format = pyaudio.paInt16 # 데이터 형식
channels = 1
rate = 16000 # 샘플링 레이트
chunk = 1600 # 블록 크기(0.1초 간격)
audio = pyaudio.PyAudio() # PyAudio 시작

# 스트리밍 시작
stream = audio.open(format=format, channels=channels,
                    rate=rate, input=True,
                    frames_per_buffer=chunk) 
buffer_class = buffer(stream, chunk, rate, n_mfcc, stream_save)
multithread = threading.Thread(target = buffer_class.stream_start)
multithread.start()

# LSTM 모델 불러오기
model = ModifiedLSTM(class_list_want).to(device)  # 먼저 모델 객체를 생성
model.load_state_dict(torch.load(model_path))  # 저장된 모델 파라미터를 로드
model.eval()  # 평가 모드로 설정
print('모델 불러오기 완료')
while True:
    if len(buffer_class.buffer) > 10:
        mfcc = buffer_class.popleft()
        dataset = AudioDataset(mfcc)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=1)
        for inputs in data_loader:
            inputs = inputs.to(device)
        outputs = model(inputs)
        probabilities = F.softmax(outputs, dim=1) # Softmax 함수를 적용하여 확률 분포를 얻음
        max_prob, pred = torch.max(probabilities, 1)  # 최대 확률과 해당 인덱스
        class_name = class_list_want[int(pred)] 
        conf = round(max_prob.item(), 3)  # 확률 값을 Python float으로 변환
        if not class_name == 'noise':
            print(f'{class_name}:{conf}')