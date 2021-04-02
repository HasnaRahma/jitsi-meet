#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler,normalize
from sklearn.cluster import estimate_bandwidth
(#les, algos, de, classification, non-supervisée)
import scipy.cluster.hierarchy as shc
from sklearn.cluster import MeanShift
from sklearn.mixture import GaussianMixture
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')


# In[2]:


DS = pd.read_csv('D:/Université Lorraine/Synthèse/Task2_CC GENERAL.csv')
DS.describe()


# In[9]:


print(' Ce data set contient {} lignes et {} colonne.\n'.format(DS.shape[0],DS.shape[1]))
DS.info()


# In[5]:


#Il faut préparer le Dataset en éliminant toutes les valeurs null
def null_values(DS):
    nv=pd.DataFrame(DS.isnull().sum()).rename(columns={0:'Missing_Records'})
    return nv[nv.Missing_Records>0].sort_values('Missing_Records', ascending=False)
null_values(DS)


# In[6]:


DS['MINIMUM_PAYMENTS']=DS['MINIMUM_PAYMENTS'].fillna(DS.MINIMUM_PAYMENTS.mean())
DS['CREDIT_LIMIT']=DS['CREDIT_LIMIT'].fillna(DS.CREDIT_LIMIT.mean())
null_values(DS).sum()


# In[10]:


#Supprimer la colonne qualitative
DS=DS.drop('CUST_ID', axis=1)


# In[11]:



# There are lots of outliers in columns but we will not apply winsorize or another methods to them.Because we may have information loss.
# They may represent another clusters.
Q1 = DS.quantile(0.25)
Q3 = DS.quantile(0.75)
IQR = Q3 - Q1
((DS[DS.columns ]< (Q1 - 1.5 * IQR)) | (DS[DS.columns] > (Q3 + 1.5 * IQR))).sum()


# In[12]:


# StandardScaler nous permet de faire la Standardisation, c'est indisponsable car en général, 
#un ensemble de données contient des variables d'échelle différente.
scaler=StandardScaler()
DS_scl=scaler.fit_transform(DS)
#Normalisation des données
norm=normalize(DS_scl)
DS_norm=pd.DataFrame(norm)


# In[13]:


est_bandwidth = estimate_bandwidth(DS_norm,quantile=0.1,n_samples=10000)
mean_shift = MeanShift(bandwidth= est_bandwidth, bin_seeding=True).fit(DS_norm)
labels_unique=np.unique(mean_shift.labels_)
n_clusters_=len(labels_unique)
print("Nombre de clusters esimé : %d" % n_clusters_)


# In[14]:



print('Silhouette Score for MeanShift:'+str(metrics.silhouette_score(DS_norm,mean_shift.labels_,metric='euclidean').round(3)))
print('Davies Bouldin Score for MeanShift:'+str(metrics.davies_bouldin_score(DS_norm,mean_shift.labels_).round(3)))


# In[15]:


#On affiche les résultats dans un graphe de 3 dimensions mais il faut appliquer ACP pour avoir le résultat final
pca = PCA(n_components=3).fit_transform(DS_norm)
fig = plt.figure(figsize=(12, 7), dpi=80, facecolor='w', edgecolor='k')
ax = plt.axes(projection="3d")
ax.scatter3D(pca.T[0],pca.T[1],pca.T[2],c=mean_shift.labels_, cmap='Spectral')

xLabel = ax.set_xlabel('X')
yLabel = ax.set_ylabel('Y')
zLabel = ax.set_zlabel('Z')


# In[ ]:


#modele = MeanShift(bandwidth= est_bandwidth, bin_seeding=True)
#modele.fit(DS_norm)
#modele.predict(DS_norm)
#ax.scatter(DS_norm.iloc[:,0], DS_norm.iloc[:,1], c = modele.fit(DS_norm))


# In[ ]:





# In[16]:


models = [GaussianMixture(n,covariance_type='tied', random_state=123).fit(DS_norm) for n in range(2,15)]
plt.plot(range(2,15), [m.bic(DS_norm) for m in models], label='BIC')
plt.plot(range(2,15), [m.aic(DS_norm) for m in models], label='AIC')
plt.legend()
plt.xlabel('n_components')
plt.show()


# In[17]:


parameters=['full','tied','diag','spherical']
n_clusters=np.arange(1,21)
results_=pd.DataFrame(columns=['Covariance Type','Number of Cluster','Silhouette Score','Davies Bouldin Score'])
for i in parameters:
    for j in n_clusters:
        gmm_cluster=GaussianMixture(n_components=j,covariance_type=i,random_state=123)
        clusters=gmm_cluster.fit_predict(DS_norm
)
        if len(np.unique(clusters))>=2:
          results_=results_.append({"Covariance Type":i,'Number of Cluster':j,"Silhouette Score":metrics.silhouette_score(DS_norm
,clusters),
                                    'Davies Bouldin Score':metrics.davies_bouldin_score(DS_norm,clusters)}
                                   ,ignore_index=True)


# In[18]:


display(results_.sort_values(by=["Silhouette Score"], ascending=False)[:5])


# In[22]:


gmm_cluster=GaussianMixture(n_components=5,covariance_type="spherical",random_state=123)
gmm_cluster.fit(DS_norm)
gmm_labels = gmm_cluster.predict(DS_norm)


# In[23]:


fig = plt.figure(figsize=(12, 7), dpi=80, facecolor='w', edgecolor='k')
ax = plt.axes(projection="3d")
ax.scatter3D(pca.T[0],pca.T[1],pca.T[2],c=gmm_labels,cmap='Spectral')

xLabel = ax.set_xlabel('X')
yLabel = ax.set_ylabel('Y')
zLabel = ax.set_zlabel('Z')


# In[21]:


#DBSCAN
results=pd.DataFrame(columns=['Eps','Min_Samples','Number of Cluster','Silhouette Score'])
for i in range(1,12):
  for j in range(1,12):
      dbscan_cluster = DBSCAN(eps=i*0.2, min_samples=j)
      clusters=dbscan_cluster.fit_predict(DS_norm)
      if len(np.unique(clusters))>2:
          results=results.append({'Eps':i*0.2,
                        'Min_Samples':j,
                        'Number of Cluster':len(np.unique(clusters)),
                        'Silhouette Score':metrics.silhouette_score(DS_norm,clusters),
                        'Davies Bouldin Score':metrics.davies_bouldin_score(DS_norm,clusters)}, ignore_index=True)


# In[24]:


results.sort_values('Silhouette Score',ascending=False)[:5]


# In[25]:


dbscan_cluster = DBSCAN(eps=0.4, min_samples=4)
db_clusters=dbscan_cluster.fit_predict(DS_norm)


# In[26]:


fig = plt.figure(figsize=(12, 7), dpi=80, facecolor='w', edgecolor='k')
ax = plt.axes(projection="3d")
ax.scatter3D(pca.T[0],pca.T[1],pca.T[2],c=db_clusters,cmap='Spectral')

xLabel = ax.set_xlabel('X')
yLabel = ax.set_ylabel('Y')
zLabel = ax.set_zlabel('Z')


# In[27]:


algorithms=["DBSCAN","Gaussian Mixture Model","MeanShift"]

# Silhouette Score
ss=[metrics.silhouette_score(DS_norm,db_clusters),metrics.silhouette_score(DS_norm,gmm_labels),metrics.silhouette_score(DS_norm,mean_shift.labels_)]

# Davies Bouldin Score
db=[metrics.davies_bouldin_score(DS_norm,db_clusters),metrics.davies_bouldin_score(DS_norm,gmm_labels),metrics.davies_bouldin_score(DS_norm,mean_shift.labels_)]


# In[28]:


comprsn={"Algorithms":algorithms,"Davies Bouldin":db,"Silhouette Score":ss}
compdf=pd.DataFrame(comprsn)
display(compdf.sort_values(by=["Silhouette Score"], ascending=False))


# In[29]:


DS['Clusters']=list(gmm_labels)
customers=pd.DataFrame(DS['Clusters'].value_counts()).rename(columns={'Clusters':'Nombre de Clusters'})
customers.T


# In[30]:


means=pd.DataFrame(DS.describe().loc['mean'])
means.T.iloc[:,[0,1,6,8,9,11,12,16]].round(1)


# In[32]:


DS.set_index('Clusters')
grouped=DS.groupby(by='Clusters').mean().round(1)
grouped.iloc[:,[0,1,6,8,9,11,12,16]]


# In[33]:


features=["BALANCE","BALANCE_FREQUENCY","PURCHASES_FREQUENCY","PURCHASES_INSTALLMENTS_FREQUENCY","CASH_ADVANCE_FREQUENCY","PURCHASES_TRX","CREDIT_LIMIT","TENURE"]
plt.figure(figsize=(15,10))
for i,j in enumerate(features):
    plt.subplot(3,3,i+1)
    sns.barplot(grouped.index,grouped[j])
    plt.title(j,fontdict={'color':'darkblue'})
plt.tight_layout()
plt.show()


# In[ ]:




