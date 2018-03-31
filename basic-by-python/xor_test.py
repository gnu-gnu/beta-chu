# pandas가 제공하는 데이터 가공을 이용해서 출력하기
import pandas as pd
from sklearn import svm, metrics

#실제 데이터
xor_input = [
    [0,0,0],
    [0,1,1],
    [1,0,1],
    [1,1,0]
]

xor_df = pd.DataFrame(xor_input) #pandas dataframe -> 2차원의 자료구조로 만듬 (행렬)
xor_data = xor_df.ix[:,0:1] #모든행에 대해서, [0]~[1] -> data
xor_label = xor_df.ix[:,2] #모든행에 대해서 [2] -> label

clf = svm.SVC()
clf.fit(xor_data, xor_label)
pre = clf.predict(xor_data)
print(pre)

ac_score = metrics.accuracy_score(xor_label, pre)
print("정답율 : {0:2}".format(ac_score))