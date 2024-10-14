import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier

import warnings
warnings.filterwarnings('ignore')

random_state = 42

n_samples = 20000
n_features = 15
n_calsses = 2
noise_moon = 0.5
noise_circle = 0.5
noise_class = 0.5

x,y = make_classification(n_samples=n_samples,
                          n_features=n_features,
                          n_repeated=0, # Veri kümesinde tekrarlanan özellik olmayacak.
                          n_redundant=0, # Veri kümesinde diğer özelliklerden türetilmiş özellik olmayacak.
                          n_informative=n_features-1, # Bu özelliklerin 9'u doğrudan sınıflar arasında ayrım yapacak bilgi içerir.
                          random_state=random_state,
                          n_clusters_per_class=1, # Her sınıf için kaç tane küme olacağını belirtir.
                          flip_y=noise_class) # Etiketlerin bir kısmını yanlış sınıfa çevirmek için kullanılan bir oran belirtir.

data = pd.DataFrame(x)
data['target']=y
plt.figure()
sns.scatterplot(x = data.iloc[:,0], y = data.iloc[:,1], hue='target', data=data)

data_classification = (x,y)

moon = make_moons(n_samples=n_samples,
                 noise = noise_moon, #Verilere %30 gürültü eklenerek örneklerin pozisyonları rastgele değiştirilecek
                 random_state=random_state)

# data = pd.DataFrame(moon[0])
# data['target']=moon[1]
# plt.figure()
# sns.scatterplot(x = data.iloc[:,0], y = data.iloc[:,1], hue='target', data=data)

circle = make_circles(n_samples=n_samples, 
                      factor=0.1, # İç dairenin yarıçapını, dış daireye göre oranını belirtir.
                      noise=noise_circle, 
                      random_state=random_state)

# data = pd.DataFrame(circle[0])
# data['target']=circle[1]
# plt.figure()
# sns.scatterplot(x = data.iloc[:,0], y = data.iloc[:,1], hue='target', data=data)

datasets = [moon, circle] #moon ve circle veri setlerini tutan bir liste.

#%% KNN, SVM, DT

n_estimators = 10 

svc = SVC()
knn = KNeighborsClassifier(n_neighbors=15)
dt = DecisionTreeClassifier(random_state=random_state, max_depth=2) #max_depth Kök düğümden başlayıp iki seviye aşağıya inebilir
rf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state, max_depth=2)
#n_estimators : agac sayisi
ada = AdaBoostClassifier(base_estimator = dt, n_estimators=n_estimators, random_state=random_state)
v1 = VotingClassifier(estimators = [('svc', svc),('knn', knn),('dt', dt),('rf', rf),('ada', ada)])

names = ["SVC", "KNN", "Decision Tree", "Random Forest", "AdaBoost", "V1"]
classifiers = [svc, knn, dt, rf, ada, v1]

h=0.2
i=1

figure = plt.figure(figsize=(18,6))
for ds_cnt, ds in enumerate(datasets): #index, oge
    x,y = ds
    x=RobustScaler().fit_transform(x)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=random_state)
    
    x_min, x_max = x[:,0].min()-.5, x[:,0].max()+.5
    y_min, y_max = x[:,1].min()-.5, x[:,1].max()+.5
    xx, yy = np.meshgrid(np.arange(x_min, y_max,h),
                         np.arange(y_min, y_max,h))
    cm=plt.cm.RdBu
    cm_bright = ListedColormap(['#FF0000','#0000FF'])
    
    ax = plt.subplot(len(datasets),len(classifiers)+1, i )
    if ds_cnt ==0:
        ax.set_title('Input Data')
        
    ax.scatter(x_train[:,0], x_train[:,1], c=y_train, cmap = cm_bright, edgecolors='k')
    ax.scatter(x_test[:,0], x_test[:,1], c=y_test, cmap=cm_bright, alpha=0.6, marker='^', edgecolors='k')
    
    ax.set_xticks(())
    ax.set_yticks(())
    i+=1
    print('Dataset # {}'.format(ds_cnt))

    for name, clf in zip(names, classifiers):
        ax = plt.subplot(len(datasets), len(classifiers)+1, i)
        clf.fit(x_train, y_train)
        score = clf.score(x_test, y_test)
        print('{}: test set score: {}'.format(name, score))
        score_train = clf.score(x_train, y_train)
        print('{}: train set score: {}'.format(name, score_train))
        print()

        #Her sınıflandırıcı için karar sınırlarını (decision boundary) görselleştirmek amacıyla, bir meshgrid oluşturuluyor. Bu grid üzerinde sınıflandırıcının tahminlerine göre zemin renklendirmesi (contourf) yapılıyor.
        if hasattr(clf, "desicion_function"): #Eğer sınıflandırıcının decision_function fonksiyonu varsa (SVM gibi), bu fonksiyon kullanılıyor, yoksa predict() fonksiyonu çağrılıyor.
            z = clf.desicion_function(np.c_[xx.ravel(), yy.ravel()]) #decision_function, bir örneğin hangi sınıfa ait olduğuna karar verirken, her sınıfa olan uzaklığını veya olasılığını gösteren puanları döndürür.
            #c_ : iki veya daha fazla diziyi (array) yan yana birleştirmek için kullanılır
            #ravel() : her bir koordinatın x ve y değerlerini düzleştirir.
            #hasattr() : bir nesnenin belirli bir niteliğe (attribute) sahip olup olmadığını kontrol eder. 
        else:
            z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

        z = z.reshape(xx.shape)
        ax.contourf(xx, yy, z, cmap = cm, alpha=.8)

        ax.scatter(x_train[:,0], x_train[:,1], c=y_train, cmap = cm_bright, edgecolors='k')
        ax.scatter(x_test[:,0], x_test[:,1], c=y_test, cmap=cm_bright, alpha=0.6, marker='^', edgecolors='white')
        ax.set_xticks(())
        ax.set_yticks(())
        if ds_cnt==0:
            ax.set_title(name)
        score = score*100
        ax.text(xx.max()-.3, yy.min()+.3, ('%.1f' %score), size = 15, horizontalalignment='right')
        i+=1
    print('--------------------------------------------')

plt.tight_layout()
plt.show()

def make_classify(dc, clf, name):
    x, y = dc
    x = RobustScaler().fit_transform(x)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.4, random_state=random_state)
    
    for name, clf in zip(names, classifiers):
        clf.fit(x_train, y_train)
        score = clf.score(x_test,y_test)
        print("{}: test set score: {} ".format(name, score))
        score_train = clf.score(x_train, y_train)  
        print("{}: train set score: {} ".format(name, score_train))
        print()
        
print("Dataset # 2")   
make_classify(data_classification, classifiers, names)  
      

  
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    



