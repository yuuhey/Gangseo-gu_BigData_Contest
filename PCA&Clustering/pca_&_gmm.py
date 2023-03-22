# -*- coding: utf-8 -*-

# 코랩 한글 깨지는거
!sudo apt-get install -y fonts-nanum
!sudo fc-cache -fv
!rm ~/.cache/matplotlib -rf

import matplotlib.pyplot as plt

plt.rc('font', family='NanumBarunGothic')

# Commented out IPython magic to ensure Python compatibility.
# 그래프 한글 안깨지게
import seaborn as sns

!sudo apt-get install -y fonts-nanum
!sudo fc-cache -fv
!rm ~/.cache/matplotlib -rf


import matplotlib.pyplot as plt

# 브라우저에서 바로 그려지도록 
# %matplotlib inline

# 그래프에 retina display 적용
# %config InlineBackend.figure_format = 'retina'

# Colab 의 한글 폰트 설정
plt.rc('font', family='NanumBarunGothic') 

# 유니코드에서  음수 부호설정
plt.rc('axes', unicode_minus=False)

import warnings
warnings.filterwarnings(action='ignore')

# Commented out IPython magic to ensure Python compatibility.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# %matplotlib inline

a = pd.read_csv("merge_target.csv", index_col = 0)

a = a.fillna(0)

# drop_list 제외 df 선택
drop_list = ['혼합학급수', '특수학급수', '혼합유아수','학급수','만3~5세유아수', '만3~5세학급수','일반수급자 기초생활수급권자수', '시설수급자 기초생활수급권자수','외국인','귀화자','18세미만 차상위수급권자수']
a.drop(drop_list, inplace=True, axis=1)

a

"""## 표준화

standard scaler 썼다가 데이터가 적고 noise가 있어 robust scaler로 바꿈 -> PCA에서 주성분의 설명력이 더 좋아짐
"""

from sklearn.preprocessing import RobustScaler

x = a.drop(["키즈카페수","행정동"], axis=1).values # 독립변인들의 value값만 추출
y = a['키즈카페수'].values # 종속변인 추출

x = RobustScaler().fit_transform(a.drop(["키즈카페수","행정동"], axis=1)) # x객체에 x를 표준화한 데이터를 저장

x = pd.DataFrame(x, columns=a.drop(["키즈카페수","행정동"], axis=1).columns)
x

"""```
# 정규화 코드
from sklearn.preprocessing import Normalizer

x = Normalizer().fit_transform(a.drop(["키즈카페수","행정동"], axis=1)) # x객체에 x를 표준화한 데이터를 저장

x = pd.DataFrame(x, columns=a.drop(["키즈카페수","행정동"], axis=1).columns)
x
```

정규화 여부에 따른 설명력 확인 결과 정규화를 안 하는 것이 더 높은 분산 설명력을 가져 정규화를 하지 않음
"""

x.columns

"""## PCA"""

from sklearn.decomposition import PCA
pca = PCA(random_state=1107)
printcipalComponents = pca.fit_transform(x)
pd.Series(np.cumsum(pca.explained_variance_ratio_))

"""제4주성분까지 사용하여야 80% 이상의 설명력을 가짐"""

# eigenvector per each PC
data_pca5 = pd.DataFrame(pca.components_[0:4],
                         columns=a.drop(["키즈카페수","행정동"], axis=1).columns,
                         index = ['PC1','PC2','PC3', 'PC4']).T

data_pca5

"""각 주성분에서 변수들의 기여도 확인 결과 

PC1 -> 결혼이민자

PC2 -> 유아수

PC3 -> 기초생활수급자수, 지하철역 수

PC4 -> 주차장면수, 지하철역수

언급한 변수들이 많이 작용
"""

# 주성분 벡터 시각화
import seaborn as sns

plt.figure(figsize=(14, 10))
sns.heatmap(data_pca5,
            annot=True,
            cmap='RdYlGn',
            cbar_kws={'shrink' : 0.5}           
           )

plt.show()

plt.rcParams['figure.figsize'] = (7, 7)
plt.plot(range(1, a.drop(["키즈카페수","행정동"], axis=1).shape[1]+1), pca.explained_variance_ratio_)
plt.xlabel("number of Principal Components", fontsize=12)
plt.ylabel("% of Variance Explained", fontsize=12)
plt.ylim(0,1)
plt.show()

pca = PCA(n_components = 4, random_state = 1107)

pca.fit(x)
data_pca = pd.DataFrame(pca.transform(x), 
                        columns = (["PC1", "PC2", "PC3", "PC4"]))

data_pca.describe().T

data_pca

"""## GMM"""

from sklearn.mixture import GaussianMixture

n_components = np.arange(1, 11)
models = [GaussianMixture(n, covariance_type='full', random_state=42).fit(data_pca) for n in n_components]
plt.plot(n_components, [m.bic(data_pca) for m in models], label='BIC')
plt.plot(n_components, [m.aic(data_pca) for m in models], label='AIC')
plt.legend(loc='best')
plt.xlabel('n_components');

"""GMM은 K-means와 달리 AIC와 BIC로 성능 평가 -> 낮아야 좋음

쩝...
"""

gmm = GaussianMixture(n_components=3, random_state=1107)
gmm_labels = gmm.fit_predict(data_pca)

data_pca['gmm_cluster'] = gmm_labels

# print(data_pca.groupby('키즈카페수')['gmm_cluster'].value_counts())



name = pd.concat([a['행정동'], data_pca['gmm_cluster']], axis=1)
name

# 군집별 행정동 개수
name['gmm_cluster'].value_counts()

# 군집별 행정동
for i in range(0,3):
  lst = list(name[name['gmm_cluster']==i]['행정동'])
  print('cluster_',i, lst)

"""### 시각화"""

# 2 dimension

plt.figure(figsize=(8,6))

sns.scatterplot(data = data_pca, x = 'PC1', y='PC2', hue='gmm_cluster')
plt.title('The Plot Of The Clusters(2D)')
plt.show()

# 3 dimension

x = data_pca['PC1']
y = data_pca['PC2']
z = data_pca['PC3']

fig = plt.figure(figsize=(12,10))
ax = plt.subplot(111, projection='3d')
ax.scatter(x, y, z, s=40, c=data_pca["gmm_cluster"], marker='o', alpha = 0.5, cmap = 'Spectral')
ax.set_title("The Plot Of The Clusters(3D)")
ax.set_xlabel("x")
ax.set_ylabel("y")
plt.show()

"""유의미한 결과가 나오지 않아 해석 불가..

'p_l','sub_num','정류소수'

'유치원개수', '특수유아수', '유아수', '어린이집_어린이수', '0~9세_아동수', '초등학교명', 

'전체 기초생활수급자수', '전체 차상위수급권자수'

'결혼이민자','귀화 및 외국국적 자녀','국내출생'
"""
