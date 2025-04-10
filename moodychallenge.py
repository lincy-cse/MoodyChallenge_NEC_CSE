import pandas as pd
df1=pd.read_csv('/content/drive/MyDrive/Colab Notebooks/Moody Challenge/Chagas1.csv')
df2=pd.read_csv('/content/drive/MyDrive/Colab Notebooks/Moody Challenge/ChagasLabel.csv')
merged_df = pd.merge(df1, df2, on='exam_id',how='left')
merged_df

merged_df["chagas"].fillna(False,inplace=True)
merged_df

import matplotlib.pyplot as plt
merged_df.hist(figsize=(10,10))
plt.show()

train_X=merged_df.iloc[0:1000,0:7]
train_Y=merged_df.iloc[0:1000,7]
test_X=merged_df.iloc[1001:1610,0:7]
test_Y=merged_df.iloc[1001:1610,7]

train_X.head()
train_Y.head()
test_X.head()
test_Y.head()

#SVM
from sklearn import svm
model=svm.SVC(kernel='linear')
model.fit(train_X,train_Y)

predict=model.predict(test_X)
from sklearn import metrics
print("acuracy:", metrics.accuracy_score(test_Y,y_pred=predict))

