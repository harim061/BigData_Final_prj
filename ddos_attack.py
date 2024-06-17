# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 16:49:30 2024

@author: doris
"""
import warnings
warnings.filterwarnings(action='ignore') 

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix, matthews_corrcoef
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA



#%%

# 1) CIC-IDS dataset 불러오기

file_path = './CIC-IDS- 2017/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv'

df = pd.read_csv(file_path)

df.head()

#%%

df.info()

#%%

# 2) 전처리하기

# 2-1) 열 이름 전처리하기 (빈칸 삭제)
df.columns = df.columns.str.strip()

print(df.shape[0])

#%%

# label 열 > ddos / benign(정상)
df['Label'].value_counts()

#%%

# 2-2) 
# 결측값 있는지 확인
null = df.isnull().sum()
null[null > 0].index.tolist()

# 결과 ['Flow Bytes/s'] 에 결측값 존재 
# 데이터 흐름의 초당 바이트 전송량

#%%

# 결측값 처리하기 > 평균값으로 대체하기
df['Flow Bytes/s'] = pd.to_numeric(df['Flow Bytes/s'], errors='coerce')

mean_value = df['Flow Bytes/s'].mean()
df['Flow Bytes/s'].fillna(mean_value,inplace=True)

# 다시 확인
null = df.isnull().sum()
null[null>0].index.tolist()

# [] > 처리 완료 

#%%

# 2-3) 무한대 값 처리 
df.replace([np.inf, -np.inf], np.nan, inplace=True)


# 2-4 ) dropping nan values
df.dropna(inplace=True)
df.shape


#%%

# 2-5) 레이블 인코딩

df['Label'] = df['Label'].apply(lambda x: 1 if x == 'DDoS' else 0)

#%%

# 레이블 시각화

plt.rcParams.update({'font.size': 14})
# 레이블 카운트
label_counts = df['Label'].value_counts().sort_index()

# 그래프 그리기
plt.figure(figsize=(10, 6))
ax = sns.barplot(x=label_counts.index, y=label_counts.values)

# 레이블 설정
plt.title('Class Distribution')
plt.xlabel('Label (0: benign, 1: DDoS)')
plt.ylabel('Count')

# 막대 위에 숫자 표시
for i, v in enumerate(label_counts.values):
    ax.text(i, v + 0.1, str(v), ha='center', va='bottom')

# x축 레이블 설정
plt.xticks([0, 1], ['Benign (0)', 'DDoS (1)'])

plt.tight_layout()
plt.show()


#%%

# 2-6) int로 type 변경
df=df.astype(int)


#%%

# 3) SelectKBest로 열 선택하기
# 특성과 레이블 분리
X = df.drop(columns=['Label'])
y = df['Label']

# 피처로 사용할 데이터에 대해 정규 분포 스케일링 수행
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#%%

num_columns = df.shape[1]

# k 값 설정 (선택할 특성의 수)
k = min(10,num_columns)

# selectKBest 
k_best = SelectKBest(score_func =f_classif, k=k)

# 정규화된 데이터에 대해 k개 선택
X_new = k_best.fit_transform(X_scaled,y)

#%%
# 선택된 특징 확인하기
selected_indices = k_best.get_support()
selected_features = X.columns[selected_indices]
selected_features

print(selected_features)


#%%

df_new = X[selected_features]
df_new['Label'] = df['Label']
df_new

#%%

#  3-2) 레이블별 특성 분포 시각화

for feature in selected_features:
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    df_new[df_new['Label'] == 0][feature].hist(bins=30, alpha=0.5, color='b', label='Benign')
    df_new[df_new['Label'] == 1][feature].hist(bins=30, alpha=0.5, color='r', label='DDoS')
    plt.title(f"{feature} Distribution by Label")
    plt.xlabel(feature)
    plt.ylabel("Frequency")
    plt.legend()

    plt.subplot(1, 2, 2)
    sns.boxplot(x='Label', y=feature, data=df_new)
    plt.title(f"{feature} Boxplot by Label")
    plt.show()

#%% 

# 4) 상관관계 분석하기
plt.figure(figsize=(12, 10))
corr_matrix = df_new.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Selected Features')
plt.show()


#%%

# 5) 학습

# 특성과 레이블 분리
X = df_new.drop(columns=['Label'])
y = df_new['Label']

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 랜덤 포레스트 모델 학습
model = RandomForestClassifier(random_state=42)

model.fit(X_train, y_train)

# 예측
y_pred = model.predict(X_test)



#%%

# 피처 중요도 확인

import matplotlib.pyplot as plt
import seaborn as sns

# 피처 중요도 계산
feature_importances = model.feature_importances_

# 피처 중요도를 데이터프레임으로 변환
feature_importances_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': feature_importances
})

# 중요도 순으로 정렬
feature_importances_df = feature_importances_df.sort_values(by='Importance', ascending=False)

# 피처 중요도 시각화
plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=feature_importances_df)
plt.title('Feature Importances')
plt.show()

#%%

# 6) 평가하기

accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
mcc = matthews_corrcoef(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("Accuracy:", accuracy)
print("Recall:", recall)
print("Precision:", precision)
print("F1 Score:", f1)
print("MCC:", mcc)
print("Confusion Matrix:\n", cm)


#%%

# 7) 성능 지표 시각화

# 성능 지표 시각화
metrics = ['Accuracy', 'Recall', 'Precision', 'F1 Score', 'MCC']
scores = [accuracy, recall, precision, f1, mcc]

x = np.arange(len(metrics))
width = 0.5

fig, ax = plt.subplots(figsize=(10, 10))
rects = ax.bar(x, scores, width, label='RandomForest')

ax.set_ylabel('Scores')
ax.set_title('Model Performance')
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.legend()

def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate('%.2f' % height,
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), 
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(rects)
plt.ylim(0, 1)
plt.show()

#%%

# 7-2) 혼동 행렬 시각화
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()


#%%

# 8) 나만의 데이터 넣기

new_ddos_sample = np.array([[80, 11495, 2201.8, 4795.6, 0, 427.5, 4114.5, 0, 872.3, 2901.6]])
new_benign_sample = np.array([[9282,7,0,0,6,22,14,0,34,0]])

# 예측
ddos_prediction = model.predict(new_ddos_sample)
benign_predcition = model.predict(new_benign_sample)

# 결과 출력
if ddos_prediction[0] == 1:
    print("샘플은 DDoS 공격으로 예측됩니다.")
else:
    print("샘플은 정상 트래픽으로 예측됩니다.")
    
# 결과 출력
if benign_predcition[0] == 1:
    print("샘플은 DDoS 공격으로 예측됩니다.")
else:
    print("샘플은 정상 트래픽으로 예측됩니다.")


#%%

# PCA를 사용한 시각화 준비
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# 새로운 데이터와 정상 트래픽, DDoS 트래픽 시각화
new_ddos_sample = np.array([[80, 11495, 2201.8, 4795.6, 0, 427.5, 4114.5, 0, 872.3, 2901.6]])
new_benign_sample = np.array([[9282, 7, 0, 0, 6, 22, 14, 0, 34, 0]])

# PCA 변환
new_ddos_pca = pca.transform(new_ddos_sample)
new_benign_pca = pca.transform(new_benign_sample)

# 시각화
plt.figure(figsize=(12, 8))

# 전체 데이터 산점도
plt.scatter(X_pca[y == 0, 0], X_pca[y == 0, 1], color='blue', label='Benign', alpha=0.5)
plt.scatter(X_pca[y == 1, 0], X_pca[y == 1, 1], color='red', label='DDoS', alpha=0.5)

# 새로운 데이터 표시
plt.scatter(new_ddos_pca[:, 0], new_ddos_pca[:, 1], color='green', label='New DDoS', marker='x', s=200)
plt.scatter(new_benign_pca[:, 0], new_benign_pca[:, 1], color='yellow', label='New Benign', marker='o', s=200)

plt.title('PCA Visualization of DDoS and Benign Traffic')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.show()

