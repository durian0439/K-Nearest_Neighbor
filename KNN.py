import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

df = pd.read_csv('Social_Network_Ads.csv')

df.head()

X = df.iloc[:, [2,3]]

X.head()

y = df['Purchased']



from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X = sc.fit_transform(X)

X





from sklearn.model_selection import train_test_split

X_train, X_test, y_train,y_test = train_test_split(X,y,test_size = 0.2, random_state = 0 )

X.shape

X_train.shape

X_test.shape

from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier(n_neighbors=5, metric = 'minkowski')

classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

y_pred

y_test = y_test.values

y_test



from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

cm

####  정확도

(55+21)/cm.sum()

sb.heatmap(cm, annot = True , cmap = 'RdPu', linewidths=0.5)
plt.show()



# 어떻게 나타내고 코드를 짜는지는 모르겠지만 배웠을 때 어떻게 나눠졌는지 볼 수 있던 코드
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, 
                               stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, 
                               stop = X_set[:, 1].max() + 1, step = 0.01))
plt.figure(figsize=[10,7])
plt.contourf(X1, X2, classifier.predict(
            np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Classifier (Test set)')
plt.legend()
plt.show()