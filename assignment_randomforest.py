from sklearn.metrics import confusion_matrix
import os
from openpyxl import load_workbook
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

address = 'C:/Users/박영신/OneDrive/바탕 화면/이동통신보안_과제/학습데이터'

file_list = os.listdir(address)

# 모든 파일 안의 line 한줄 한줄에서 필요한 데이터만 받아와서 리스트에 넣기 (IMSI, BEARER CREATED TIME)
data = []
for i in file_list:
    file = address + '/' + i
    f = open(file, 'r')
    
    for line in f:         
        
        line = line.split(',')
        linedata = []
        
        linedata.append(line[5])
        tempstr = line[40][11:26]
        linedata.append(tempstr)
        data.append(linedata)
    f.close()
data.sort(key = lambda x: x[0])

# [[IMSI1, [TIME1, TIME2, TIME3, ...]], [IMSI2, [TIME1, TIME2, TIME3, ...]], ...]
# 이런 형식으로 만들기 위해 IMSI와 시간(분)이 같으면 존재하는 리스트에 TIME을 추가
# IMSI가 다르거나 IMSI는 같지만 시간(분)이 다르면 새로 [IMSI, [TIME]] 형식의 리스트를 가장 큰 dic 리스트에 추가
dic = []    
for imsi, time in data:
    found = False
    for item in dic:
        if item[0] == imsi and time[:5] == item[1][0][:5]:
            item[1].append(time)
            found = True
            break
    if not found:
        dic.append([imsi, [time]])

# 학습을 위해 정상 가입자와 공격자의 시그널링 메시지를 구분할 수 있는 라벨 추가
# 정상 가입자는 NORMAL, 공격자는 ABNORMAL
feature = []
for imsi in dic:
    if ((imsi[0].startswith('450052006'))): 
        feature.append('NORMAL')
    else:
        feature.append('ABNORMAL')

# 각 리스트의 TIME들에 대한 평균, 분산을 계산
ft = 0
for imsi in dic:
    add = []
    for j in range(0, len(imsi[1])-1):
        if (len(imsi[1]) == 1):
            break  
            
        temp1 = imsi[1][j].split(':')
        temp2 = imsi[1][j+1].split(':')
        if(temp1[1] != temp2[1]):
            add.append(float(temp2[2]) + 60 - float(temp1[2]))
        else:
            add.append(float(temp2[2]) - float(temp1[2]))
        
    count = len(imsi[1])
    if (len(imsi[1]) == 1):
        avg = 0
        var = 0
    else:
        avg = np.mean(add)
        var = np.var(add)  
        
    imsi[1].clear()
    imsi[1].append(avg)
    imsi[1].append(var)     
    imsi[1].append(count)
    imsi[1].append(feature[ft])     
    ft +=1

# IMSI, AVG(평균), VAR(분산), COUNT, FEATURE(라벨) 라는 columns를 가진 데이터 프레임 생성
df = pd.DataFrame(columns = ['IMSI', 'AVG', 'VAR', 'COUNT', 'FEATURE'])
for row in dic:
    df.loc[len(df)] = [row[0]] + [str(num) for num in row[1]]

# IMSI와 마지막 줄(FEATURE)을 제외한 값 X에 저장, y에 FEATURE 저장(예측)
X = df.iloc[:, [1, 2, 3]]
y = df.iloc[:, [-1]]

# train과 test 데이터 7:3으로 나눔
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)
print("Train Data:", X_train.shape, "Test Data:", X_test.shape)

# n_estimators, max_features, max_depth의 범위를 설정하고 훈련하여, 최적의 조합을 찾음
rf = RandomForestClassifier(random_state = 1)
param_grid = [{'n_estimators' : range(5, 50, 10), 'max_features' : range(1, 4), 'max_depth' : range(3, 5)}]
gs = GridSearchCV(estimator = rf, param_grid = param_grid, scoring = 'accuracy', cv = 5, n_jobs = -1)
gs.fit(X_train, y_train.values.ravel())

# 최적의 매개변수 조합 출력, 최적일 때의 정확도 출력
print('BEST : {0}'.format(gs.best_params_))
print('Accuracy when the BEST : {0:.2f}'.format(gs.best_score_))

# 최적의 매개변수로 훈련
best_model = gs.best_estimator_

# 최적의 모델을 사용하여 테스트 데이터에 대한 정확도 계산, 출력
score = best_model.score(X_test, y_test)
print('Accuracy in the test data : {0:.2f}'.format(score))

# 테스트 데이터에 대한 예측
y_pred = best_model.predict(X_test)

# classification report 출력
report = classification_report(y_test, y_pred)
print("Classification Report:\n", report)

# 오차 행렬 계산, 출력
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix :\n", cm)
