# 붓꽃의 품종을 예측 해 보는 매우 유명한 딥러닝 입문 예제 by 싸이킷런
from sklearn import svm, metrics
from sklearn.model_selection import train_test_split
import pandas as pd

# 꽃받침 길이 / 꽃받침 폭 / 꽃잎 길이 / 꽃잎 너비 / 품종 으로 이루어진 csv 데이터를 불러옴
csv = pd.read_csv("data/iris.csv")

# iloc 인덱서를 사용해서 1~4번째 컬럼을 데이터로, 5번째 컬럼을 label로 분리
# 컬럼명을 인덱서로 사용하려면 loc, 순서를 인덱서로 사용하려면 iloc, 둘다 섞어 쓰려면 ix
# 인덱서의 입력 형식은 [행, 열]
csv_data = csv.iloc[:, :4]
csv_label = csv.iloc[:, 4:]

# train_test_split을 이용해 훈련용데이터, 테스트용 데이터, 훈련용 라벨, 테스트용 라벨을 추출
# sklearn.model_selection.train_test_split(*arrays, **options)
# arrays -> lists, numpy arrays, scipy-sparse metrics, pandas dataframes
# test_size (optional) (float, int, none) It will remain 0.25 only if train_size is unspecified, otherwise it will complement the specified train_size
# train_size train split (float, int, none)
# random_state -> (int) random seed
# shuffle -> default : true
train_data, test_data,  train_label, test_label = train_test_split(csv_data, csv_label); #train_size={float}을 작게하면 할수록 정밀도가 낮아짐

# 학습시키기
clf = svm.SVC()
clf.fit(train_data, train_label.values.ravel()) #fit 시킬 때 1차원 배열로 flatten 시켜준다. (반대로 다차원으로 만들 때는 reshape(x,y)

# 학습 시킨 모델을 테스트 데이터로 예측하기
predict = clf.predict(test_data)
# 테스트 데이터와 예측 데이터의 비교를 통해 정확도를 예측해주는 메소드
accuracy = metrics.accuracy_score(test_label, predict)

# 자세한 결과 보기
test_label_flatten = test_label.values.ravel();
for index in range(len(test_label_flatten)):
    print(index, "{0:30} {1:30} {2:}".format(test_label_flatten[index], predict[index], test_label_flatten[index] == predict[index]))

print("Accuracy =", accuracy)


