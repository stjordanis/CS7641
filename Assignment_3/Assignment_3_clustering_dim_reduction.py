
# coding: utf-8

# In[12]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time


# ### Create synthetic data using Scikit learn `make_blob` method

# In[13]:


from sklearn.datasets import make_classification
from sklearn.datasets import make_blobs


# In[14]:


n_features = 8
n_cluster = 5
cluster_std = 2.5
n_samples = 1000


# In[15]:


data1 = make_blobs(n_samples=n_samples,n_features=n_features,
                   centers=n_cluster,cluster_std=cluster_std,random_state=101)


# In[16]:


d1 = data1[0]


# In[17]:


df1=pd.DataFrame(data=d1,columns=['Feature_'+str(i) for i in range(1,n_features+1)])
df1.head()


# In[18]:


from itertools import combinations


# In[19]:


lst_vars=list(combinations(df1.columns,2))


# In[20]:


len(lst_vars)


# In[21]:


plt.figure(figsize=(21,35))
for i in range(1,29):
    plt.subplot(7,4,i)
    dim1=lst_vars[i-1][0]
    dim2=lst_vars[i-1][1]
    plt.scatter(df1[dim1],df1[dim2],c=data1[1],edgecolor='k')
    plt.xlabel(f"{dim1}",fontsize=13)
    plt.ylabel(f"{dim2}",fontsize=13)


# In[22]:


df1.describe().transpose()


# ### How are the classes separated (boxplots)

# In[23]:


plt.figure(figsize=(21,15))
for i,c in enumerate(df1.columns):
    plt.subplot(3,3,i+1)
    sns.boxplot(y=df1[c],x=data1[1])
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlabel("Class",fontsize=15)
    plt.ylabel(c,fontsize=15)
    #plt.show()


# ## k-means clustering

# In[24]:


from sklearn.cluster import KMeans


# ### Unlabled data

# In[25]:


X=df1


# In[26]:


X.head()


# In[27]:


y=data1[1]


# ### Scaling

# In[28]:


from sklearn.preprocessing import MinMaxScaler


# In[29]:


scaler = MinMaxScaler()


# In[30]:


X_scaled=scaler.fit_transform(X)


# ### Metrics

# In[31]:


from sklearn.metrics.cluster import adjusted_mutual_info_score


# In[32]:


from sklearn.metrics import silhouette_score


# In[33]:


from sklearn.metrics import adjusted_rand_score


# In[34]:


from sklearn.metrics import completeness_score


# In[35]:


from sklearn.metrics.cluster import homogeneity_score


# In[36]:


from sklearn.metrics import v_measure_score


# ### Running k-means and computing inter-cluster distance score for various *k* values

# In[37]:


km_sse= []
km_silhouette = []
km_vmeasure =[]
km_ami = []
km_homogeneity = []
km_completeness = []

cluster_range = (2,12)

for i in range(cluster_range[0],cluster_range[1]):
    km = KMeans(n_clusters=i, random_state=0).fit(X_scaled)
    preds = km.predict(X_scaled)
    km_sse.append(-km.score(X_scaled))
    km_silhouette.append(silhouette_score(X_scaled,preds))
    km_vmeasure.append(v_measure_score(y,preds))
    km_ami.append(adjusted_mutual_info_score(y,preds))
    km_homogeneity.append(homogeneity_score(y,preds))
    km_completeness.append(completeness_score(y,preds))
    print(f"Done for cluster {i}")


# ### Plotting various cluster evaluation metrics as function of number of clusters

# In[38]:


plt.figure(figsize=(21,10))

#SSE
plt.subplot(2,3,1)
plt.plot([i for i in range(cluster_range[0],cluster_range[1])],km_sse,'b-o',linewidth=3,markersize=12)
plt.grid(True)
plt.title("SSE score vs. number of clusters",fontsize=15)
plt.xticks([i for i in range(0,cluster_range[1]+1,1)],fontsize=15)
plt.yticks(fontsize=15)

#Silhouette
plt.subplot(2,3,2)
plt.plot([i for i in range(cluster_range[0],cluster_range[1])],km_silhouette,'k-o',linewidth=3,markersize=12)
plt.grid(True)
plt.title("Silhouette score vs. number of clusters",fontsize=15)
plt.xticks([i for i in range(0,cluster_range[1]+1,1)],fontsize=15)
plt.yticks(fontsize=15)

#Homegeneity
plt.subplot(2,3,3)
plt.plot([i for i in range(cluster_range[0],cluster_range[1])],km_homogeneity,'r-o',linewidth=3,markersize=12)
plt.grid(True)
plt.title("Homegeneity score vs. number of clusters",fontsize=15)
plt.xticks([i for i in range(0,cluster_range[1]+1,1)],fontsize=15)
plt.yticks(fontsize=15)

#Completeness
plt.subplot(2,3,4)
plt.plot([i for i in range(cluster_range[0],cluster_range[1])],km_completeness,'g-o',linewidth=3,markersize=12)
plt.grid(True)
plt.title("Completeness score vs. number of clusters",fontsize=15)
plt.xticks([i for i in range(0,cluster_range[1]+1,1)],fontsize=15)
plt.yticks(fontsize=15)

#V-measure
plt.subplot(2,3,5)
plt.plot([i for i in range(cluster_range[0],cluster_range[1])],km_vmeasure,'c-o',linewidth=3,markersize=12)
plt.grid(True)
plt.title("V-measure score vs. number of clusters",fontsize=15)
plt.xticks([i for i in range(0,cluster_range[1]+1,1)],fontsize=15)
plt.yticks(fontsize=15)

#Adjusted mutual information
plt.subplot(2,3,6)
plt.plot([i for i in range(cluster_range[0],cluster_range[1])],km_ami,'m-o',linewidth=3,markersize=12)
plt.grid(True)
plt.title("Adjusted mutual information vs. number of clusters",fontsize=15)
plt.xticks([i for i in range(0,cluster_range[1]+1,1)],fontsize=15)
plt.yticks(fontsize=15)

plt.show()


# ### Creating the baseline k-means model

# In[345]:


t1=time.time()
km = KMeans(n_clusters=5,n_init=10,max_iter=500).fit(X=X_scaled)
t2=time.time()
time_km = t2-t1


# In[346]:


preds_km = km.predict(X_scaled)


# In[347]:


plt.figure(figsize=(21,35))
for i in range(1,29):
    plt.subplot(7,4,i)
    dim1=lst_vars[i-1][0]
    dim2=lst_vars[i-1][1]
    plt.scatter(df1[dim1],df1[dim2],c=preds_km,edgecolor='k')
    plt.xlabel(f"{dim1}",fontsize=13)
    plt.ylabel(f"{dim2}",fontsize=13)


# ### Estimating average running time of k-means

# In[410]:


time_km = []
for i in range(100):
    t1=time.time()
    km = KMeans(n_clusters=5,n_init=10,max_iter=500).fit(X=X_scaled)
    t2=time.time()
    delta = t2-t1
    time_km.append(delta)
time_km = np.array(time_km)
avg_time_km = time_km.mean()
print(avg_time_km)


# ### Showing that evaluation metric gets blurry as the clusters' variance increases

# In[56]:


n_features = 8
n_cluster = 5
#cluster_std = 2.5
n_samples = 1000
cluster_range = (2,12)
j = 1
for i in range(1,5):
    data1 = make_blobs(n_samples=n_samples,n_features=n_features,
                       centers=n_cluster,cluster_std=i*2,random_state=101)
    df1=pd.DataFrame(data=data1[0],columns=['Feature_'+str(i) for i in range(1,n_features+1)])
    X=df1
    X_scaled=scaler.fit_transform(X)
    km_sse=[]
    for r in range(cluster_range[0],cluster_range[1]):
        km = KMeans(n_clusters=r, random_state=0).fit(X_scaled)
        preds = km.predict(X_scaled)
        km_sse.append(-km.score(X_scaled))
    
    plt.figure(figsize=(18,6))
    plt.subplot(2,4,2*i-1)
    plt.scatter(df1['Feature_1'],df1['Feature_2'],c=data1[1],edgecolor='k')
    plt.xlabel(f"Feature_1",fontsize=13)
    plt.ylabel(f"Feature_2",fontsize=13)
    plt.grid(True)
    
    plt.subplot(2,4,2*i)
    plt.plot([i for i in range(cluster_range[0],cluster_range[1])],km_sse,'b-o',linewidth=3,markersize=12)
    plt.grid(True)
    plt.title("SSE score vs. number of clusters",fontsize=15)
    plt.xticks([i for i in range(0,cluster_range[1]+1,1)],fontsize=15)
    plt.yticks(fontsize=15)    


# ## Expectation-maximization (Gaussian Mixture Model)

# In[58]:


from sklearn.mixture import GaussianMixture


# In[61]:


n_features = 8
n_cluster = 5
cluster_std = 2.5
n_samples = 1000

data1 = make_blobs(n_samples=n_samples,n_features=n_features,
                   centers=n_cluster,cluster_std=cluster_std,random_state=101)
df1=pd.DataFrame(data=d1,columns=['Feature_'+str(i) for i in range(1,n_features+1)])
X = df1
X_scaled=scaler.fit_transform(X)


# In[62]:


gm_ll= []
gm_bic = []
gm_aic = []
gm_silhouette = []
gm_vmeasure =[]
gm_ami = []
gm_homogeneity = []
gm_completeness = []

cluster_range = (2,12)

for i in range(cluster_range[0],cluster_range[1]):
    gm = GaussianMixture(n_components=i).fit(X_scaled)
    preds = gm.predict(X_scaled)
    gm_ll.append(gm.score(X_scaled))
    gm_bic.append(-gm.bic(X_scaled))
    gm_aic.append(-gm.aic(X_scaled))
    gm_silhouette.append(silhouette_score(X_scaled,preds))
    gm_vmeasure.append(v_measure_score(y,preds))
    gm_ami.append(adjusted_mutual_info_score(y,preds))
    gm_homogeneity.append(homogeneity_score(y,preds))
    gm_completeness.append(completeness_score(y,preds))
    print(f"Done for Gaussian components {i}")


# In[63]:


plt.figure(figsize=(21,16))

#SSE
plt.subplot(3,3,1)
plt.plot([i for i in range(cluster_range[0],cluster_range[1])],gm_ll,'b-o',linewidth=3,markersize=12)
plt.grid(True)
plt.title("Log-likelihood vs. number of Gaussian centers",fontsize=15)
plt.xticks([i for i in range(0,cluster_range[1]+1,1)],fontsize=15)
plt.yticks(fontsize=15)

#Silhouette
plt.subplot(3,3,2)
plt.plot([i for i in range(cluster_range[0],cluster_range[1])],gm_silhouette,'k-o',linewidth=3,markersize=12)
plt.grid(True)
plt.title("Silhouette score vs. number of Gaussian centers",fontsize=15)
plt.xticks([i for i in range(0,cluster_range[1]+1,1)],fontsize=15)
plt.yticks(fontsize=15)

#Homegeneity
plt.subplot(3,3,3)
plt.plot([i for i in range(cluster_range[0],cluster_range[1])],gm_homogeneity,'r-o',linewidth=3,markersize=12)
plt.grid(True)
plt.title("Homegeneity score vs. number of Gaussian centers",fontsize=15)
plt.xticks([i for i in range(0,cluster_range[1]+1,1)],fontsize=15)
plt.yticks(fontsize=15)

#Completeness
plt.subplot(3,3,4)
plt.plot([i for i in range(cluster_range[0],cluster_range[1])],gm_completeness,'g-o',linewidth=3,markersize=12)
plt.grid(True)
plt.title("Completeness score vs. number of Gaussian centers",fontsize=15)
plt.xticks([i for i in range(0,cluster_range[1]+1,1)],fontsize=15)
plt.yticks(fontsize=15)

#V-measure
plt.subplot(3,3,5)
plt.plot([i for i in range(cluster_range[0],cluster_range[1])],gm_vmeasure,'c-o',linewidth=3,markersize=12)
plt.grid(True)
plt.title("V-measure score vs. number of Gaussian centers",fontsize=15)
plt.xticks([i for i in range(0,cluster_range[1]+1,1)],fontsize=15)
plt.yticks(fontsize=15)

#Adjusted mutual information
plt.subplot(3,3,6)
plt.plot([i for i in range(cluster_range[0],cluster_range[1])],gm_ami,'m-o',linewidth=3,markersize=12)
plt.grid(True)
plt.title("Adjusted mutual information vs. number of Gaussian centers",fontsize=15)
plt.xticks([i for i in range(0,cluster_range[1]+1,1)],fontsize=15)
plt.yticks(fontsize=15)

plt.show()

plt.figure(figsize=(21,5))
#BIC
plt.subplot(1,2,1)
plt.plot([i for i in range(cluster_range[0],cluster_range[1])],gm_bic,'m-o',linewidth=3,markersize=12)
plt.grid(True)
plt.title("BIC vs. number of Gaussian centers",fontsize=15)
plt.xticks([i for i in range(0,cluster_range[1]+1,1)],fontsize=15)
plt.yticks(fontsize=15)

#AIC
plt.subplot(1,2,2)
plt.plot([i for i in range(cluster_range[0],cluster_range[1])],gm_aic,'m-o',linewidth=3,markersize=12)
plt.grid(True)
plt.title("AIC vs. number of Gaussian centers",fontsize=15)
plt.xticks([i for i in range(0,cluster_range[1]+1,1)],fontsize=15)
plt.yticks(fontsize=15)

plt.show()


# ### Creating the baseline Gaussian Mixture Model

# In[351]:


t1=time.time()
gm = GaussianMixture(n_components=5,verbose=1,n_init=10,tol=1e-5,covariance_type='full',max_iter=500).fit(X_scaled)
t2=time.time()
time_gm = t2-t1


# In[352]:


gm.means_


# In[353]:


km.cluster_centers_


# In[354]:


gm.means_/km.cluster_centers_


# In[355]:


preds_gm=gm.predict(X_scaled)


# In[356]:


km_rand_score = adjusted_rand_score(preds_km,y)


# In[357]:


gm_rand_score = adjusted_rand_score(preds_gm,y)


# In[358]:


print("Adjusted Rand score for k-means",km_rand_score)
print("Adjusted Rand score for Gaussian Mixture model",gm_rand_score)


# In[359]:


plt.figure(figsize=(21,35))
for i in range(1,29):
    plt.subplot(7,4,i)
    dim1=lst_vars[i-1][0]
    dim2=lst_vars[i-1][1]
    plt.scatter(df1[dim1],df1[dim2],c=preds_gm,edgecolor='k')
    plt.xlabel(f"{dim1}",fontsize=13)
    plt.ylabel(f"{dim2}",fontsize=13)


# ### Estimating average running time of Gaussian Mixture Model

# In[413]:


time_gm = []
for i in range(100):
    t1=time.time()
    gm = GaussianMixture(n_components=5,n_init=10,tol=1e-5,
                         covariance_type='full',max_iter=500).fit(X_scaled)
    t2= time.time()
    delta = t2-t1
    time_gm.append(delta)
time_gm = np.array(time_gm)
avg_time_gm = time_gm.mean()
print(avg_time_gm)


# ## PCA

# In[78]:


from sklearn.decomposition import PCA


# In[79]:


n_prin_comp = 4


# In[80]:


pca_partial = PCA(n_components=n_prin_comp,svd_solver='full')
pca_partial.fit(X_scaled)


# In[81]:


pca_full = PCA(n_components=n_features,svd_solver='full')
pca_full.fit(X_scaled)


# ### Eigenvalues

# In[88]:


plt.figure(figsize=(8,5))
plt.title("PCA eigenvalues for Synthetic dataset", fontsize=18)
plt.plot(pca_full.explained_variance_,'b-o')
plt.grid(True)
plt.xticks(fontsize=16)
plt.xlabel("Principal Components",fontsize=15)
plt.yticks(fontsize=16)
plt.ylabel("Eigenvalues",fontsize=15)
plt.show()


# ### How much variance is explained by principal components?

# In[90]:


pca_explained_var = pca_full.explained_variance_ratio_


# In[91]:


cum_explaiend_var = pca_explained_var.cumsum()


# In[92]:


cum_explaiend_var


# In[93]:


plt.figure(figsize=(8,5))
plt.title("Scree plot - cumulative variance explained by \nprincipal components (Synthetic dataset)", fontsize=18)
plt.bar(range(1,9),cum_explaiend_var)
plt.xticks(fontsize=16)
plt.xlabel("Principal Components",fontsize=15)
plt.yticks(fontsize=16)
plt.xlim(0,9)
plt.hlines(y=0.8,xmin=0,xmax=9,linestyles='dashed',lw=3)
plt.text(x=0.5,y=0.72,s="80% variance explained",fontsize=15)
plt.hlines(y=0.9,xmin=0,xmax=9,linestyles='dashed',lw=3)
plt.text(x=0.5,y=0.92,s="90% variance explained",fontsize=15)
plt.show()


# ### Transform the original variables in principal component space and create a DataFrame

# In[368]:


X_pca = pca_partial.fit_transform(X_scaled)


# In[369]:


df_pca=pd.DataFrame(data=X_pca,columns=['Principal_comp'+str(i) for i in range(1,n_prin_comp+1)])


# ### Running k-means on the transformed features

# In[370]:


km_sse= []
km_silhouette = []
km_vmeasure =[]
km_ami = []
km_homogeneity = []
km_completeness = []

cluster_range = (2,12)

for i in range(cluster_range[0],cluster_range[1]):
    km = KMeans(n_clusters=i, random_state=0).fit(X_pca)
    preds = km.predict(X_pca)
    km_sse.append(-km.score(X_pca))
    km_silhouette.append(silhouette_score(X_pca,preds))
    km_vmeasure.append(v_measure_score(y,preds))
    km_ami.append(adjusted_mutual_info_score(y,preds))
    km_homogeneity.append(homogeneity_score(y,preds))
    km_completeness.append(completeness_score(y,preds))
    print(f"Done for cluster {i}")


# In[371]:


plt.figure(figsize=(21,11))

#SSE
plt.subplot(2,3,1)
plt.plot([i for i in range(cluster_range[0],cluster_range[1])],km_sse,'b-o',linewidth=3,markersize=12)
plt.grid(True)
plt.title("SSE score vs. number of clusters",fontsize=15)
plt.xticks([i for i in range(0,cluster_range[1]+1,1)],fontsize=15)
plt.yticks(fontsize=15)

#Silhouette
plt.subplot(2,3,2)
plt.plot([i for i in range(cluster_range[0],cluster_range[1])],km_silhouette,'k-o',linewidth=3,markersize=12)
plt.grid(True)
plt.title("Silhouette score vs. number of clusters",fontsize=15)
plt.xticks([i for i in range(0,cluster_range[1]+1,1)],fontsize=15)
plt.yticks(fontsize=15)

#Homegeneity
plt.subplot(2,3,3)
plt.plot([i for i in range(cluster_range[0],cluster_range[1])],km_homogeneity,'r-o',linewidth=3,markersize=12)
plt.grid(True)
plt.title("Homegeneity score vs. number of clusters",fontsize=15)
plt.xticks([i for i in range(0,cluster_range[1]+1,1)],fontsize=15)
plt.yticks(fontsize=15)

#Completeness
plt.subplot(2,3,4)
plt.plot([i for i in range(cluster_range[0],cluster_range[1])],km_completeness,'g-o',linewidth=3,markersize=12)
plt.grid(True)
plt.title("Completeness score vs. number of clusters",fontsize=15)
plt.xticks([i for i in range(0,cluster_range[1]+1,1)],fontsize=15)
plt.yticks(fontsize=15)

#V-measure
plt.subplot(2,3,5)
plt.plot([i for i in range(cluster_range[0],cluster_range[1])],km_vmeasure,'c-o',linewidth=3,markersize=12)
plt.grid(True)
plt.title("V-measure score vs. number of clusters",fontsize=15)
plt.xticks([i for i in range(0,cluster_range[1]+1,1)],fontsize=15)
plt.yticks(fontsize=15)

#Adjusted mutual information
plt.subplot(2,3,6)
plt.plot([i for i in range(cluster_range[0],cluster_range[1])],km_ami,'m-o',linewidth=3,markersize=12)
plt.grid(True)
plt.title("Adjusted mutual information vs. number of clusters",fontsize=15)
plt.xticks([i for i in range(0,cluster_range[1]+1,1)],fontsize=15)
plt.yticks(fontsize=15)

plt.show()


# ### K-means fitting with PCA-transformed data

# In[372]:


t1=time.time()
km_pca = KMeans(n_clusters=5,n_init=10,max_iter=500).fit(X=X_pca)
t2=time.time()
time_km_pca = t2-t1
preds_km_pca = km_pca.predict(X_pca)


# ### Visualizing the clusters after running k-means on PCA-transformed features

# In[373]:


col_pca_combi=list(combinations(df_pca.columns,2))
num_pca_combi = len(col_pca_combi)


# In[374]:


plt.figure(figsize=(21,20))
for i in range(1,num_pca_combi+1):
    plt.subplot(int(num_pca_combi/3)+1,3,i)
    dim1=col_pca_combi[i-1][0]
    dim2=col_pca_combi[i-1][1]
    plt.scatter(df_pca[dim1],df_pca[dim2],c=preds_km_pca,edgecolor='k')
    plt.xlabel(f"{dim1}",fontsize=13)
    plt.ylabel(f"{dim2}",fontsize=13)


# ### Estimating average running time of k-means with PCA

# In[414]:


time_km_pca = []
for i in range(100):
    t1=time.time()
    km_pca = KMeans(n_clusters=5,n_init=10,max_iter=500).fit(X=X_pca)
    t2= time.time()
    delta = t2-t1
    time_km_pca.append(delta)
time_km_pca = np.array(time_km_pca)
avg_time_km_pca = time_km_pca.mean()
print(avg_time_km_pca)


# ## ICA

# In[66]:


from sklearn.decomposition import FastICA


# In[67]:


n_ind_comp = 4


# In[68]:


ica_partial = FastICA(n_components=n_ind_comp)
ica_partial.fit(X_scaled)


# In[69]:


ica_full = FastICA(max_iter=1000)
ica_full.fit(X_scaled)


# In[70]:


X_ica = ica_partial.fit_transform(X_scaled)


# In[71]:


df_ica=pd.DataFrame(data=X_ica,columns=['Independent_comp'+str(i) for i in range(1,n_ind_comp+1)])


# ### Kurtosis

# In[72]:


from scipy.stats import kurtosis


# In[74]:


kurtosis(X_ica,fisher=False)


# In[94]:


X_ica_full = ica_full.fit_transform(X_scaled)


# In[100]:


kurt = kurtosis(X_ica_full,fisher=True)


# In[109]:


plt.figure(figsize=(8,5))
plt.title("ICA Kurtosis for Synthetic dataset", fontsize=18)
plt.plot(kurt,'b-o')
plt.grid(True)
plt.xticks(fontsize=16)
plt.xlabel("Independent Components",fontsize=15)
plt.yticks(fontsize=16)
plt.ylabel("Kurtosis of the transformed dataset",fontsize=15)
plt.hlines(y = 0, xmin=0, xmax=8,colors='red',linestyles='dashed')
plt.show()


# ### Running k-means on the independent features

# In[381]:


km_sse= []
km_silhouette = []
km_vmeasure =[]
km_ami = []
km_homogeneity = []
km_completeness = []

cluster_range = (2,12)

for i in range(cluster_range[0],cluster_range[1]):
    km = KMeans(n_clusters=i, random_state=0).fit(X_ica)
    preds = km.predict(X_ica)
    km_sse.append(-km.score(X_ica))
    km_silhouette.append(silhouette_score(X_ica,preds))
    km_vmeasure.append(v_measure_score(y,preds))
    km_ami.append(adjusted_mutual_info_score(y,preds))
    km_homogeneity.append(homogeneity_score(y,preds))
    km_completeness.append(completeness_score(y,preds))
    print(f"Done for cluster {i}")


# In[382]:


plt.figure(figsize=(21,11))

#SSE
plt.subplot(2,3,1)
plt.plot([i for i in range(cluster_range[0],cluster_range[1])],km_sse,'b-o',linewidth=3,markersize=12)
plt.grid(True)
plt.title("SSE score vs. number of clusters",fontsize=15)
plt.xticks([i for i in range(0,cluster_range[1]+1,1)],fontsize=15)
plt.yticks(fontsize=15)

#Silhouette
plt.subplot(2,3,2)
plt.plot([i for i in range(cluster_range[0],cluster_range[1])],km_silhouette,'k-o',linewidth=3,markersize=12)
plt.grid(True)
plt.title("Silhouette score vs. number of clusters",fontsize=15)
plt.xticks([i for i in range(0,cluster_range[1]+1,1)],fontsize=15)
plt.yticks(fontsize=15)

#Homegeneity
plt.subplot(2,3,3)
plt.plot([i for i in range(cluster_range[0],cluster_range[1])],km_homogeneity,'r-o',linewidth=3,markersize=12)
plt.grid(True)
plt.title("Homegeneity score vs. number of clusters",fontsize=15)
plt.xticks([i for i in range(0,cluster_range[1]+1,1)],fontsize=15)
plt.yticks(fontsize=15)

#Completeness
plt.subplot(2,3,4)
plt.plot([i for i in range(cluster_range[0],cluster_range[1])],km_completeness,'g-o',linewidth=3,markersize=12)
plt.grid(True)
plt.title("Completeness score vs. number of clusters",fontsize=15)
plt.xticks([i for i in range(0,cluster_range[1]+1,1)],fontsize=15)
plt.yticks(fontsize=15)

#V-measure
plt.subplot(2,3,5)
plt.plot([i for i in range(cluster_range[0],cluster_range[1])],km_vmeasure,'c-o',linewidth=3,markersize=12)
plt.grid(True)
plt.title("V-measure score vs. number of clusters",fontsize=15)
plt.xticks([i for i in range(0,cluster_range[1]+1,1)],fontsize=15)
plt.yticks(fontsize=15)

#Adjusted mutual information
plt.subplot(2,3,6)
plt.plot([i for i in range(cluster_range[0],cluster_range[1])],km_ami,'m-o',linewidth=3,markersize=12)
plt.grid(True)
plt.title("Adjusted mutual information vs. number of clusters",fontsize=15)
plt.xticks([i for i in range(0,cluster_range[1]+1,1)],fontsize=15)
plt.yticks(fontsize=15)

plt.show()


# ### K-means fitting with ICA-transformed data

# In[383]:


t1=time.time()
km_ica = KMeans(n_clusters=5,n_init=10,max_iter=500).fit(X=X_ica)
t2=time.time()
time_km_ica = t2-t1

# Predictions
preds_km_ica = km_ica.predict(X_ica)


# ### Visualizing the clusters after running k-means on ICA-transformed features

# In[384]:


col_ica_combi=list(combinations(df_ica.columns,2))
num_ica_combi = len(col_ica_combi)


# In[386]:


plt.figure(figsize=(21,20))
for i in range(1,num_ica_combi+1):
    plt.subplot(int(num_ica_combi/3)+1,3,i)
    dim1=col_ica_combi[i-1][0]
    dim2=col_ica_combi[i-1][1]
    plt.scatter(df_ica[dim1],df_ica[dim2],c=preds_km_ica,edgecolor='k')
    plt.xlabel(f"{dim1}",fontsize=13)
    plt.ylabel(f"{dim2}",fontsize=13)


# ### Estimating average running time of k-means with ICA

# In[415]:


time_km_ica = []
for i in range(100):
    t1=time.time()
    km_ica = KMeans(n_clusters=5,n_init=10,max_iter=500).fit(X=X_ica)
    t2= time.time()
    delta = t2-t1
    time_km_ica.append(delta)
time_km_ica = np.array(time_km_ica)
avg_time_km_ica = time_km_ica.mean()
print(avg_time_km_ica)


# ## Random Projection

# In[154]:


from sklearn.random_projection import GaussianRandomProjection


# In[155]:


n_random_comp = 7


# In[156]:


random_proj = GaussianRandomProjection(n_components=n_random_comp)


# In[157]:


X_random_proj = random_proj.fit_transform(X_scaled)


# In[158]:


df_random_proj=pd.DataFrame(data=X_random_proj,columns=['Random_projection'+str(i) for i in range(1,n_random_comp+1)])


# ### Running k-means on random projections

# In[159]:


km_sse= []
km_silhouette = []
km_vmeasure =[]
km_ami = []
km_homogeneity = []
km_completeness = []

cluster_range = (2,12)

for i in range(cluster_range[0],cluster_range[1]):
    km = KMeans(n_clusters=i, random_state=0).fit(X_random_proj)
    preds = km.predict(X_random_proj)
    km_sse.append(-km.score(X_random_proj))
    km_silhouette.append(silhouette_score(X_random_proj,preds))
    km_vmeasure.append(v_measure_score(y,preds))
    km_ami.append(adjusted_mutual_info_score(y,preds))
    km_homogeneity.append(homogeneity_score(y,preds))
    km_completeness.append(completeness_score(y,preds))
    print(f"Done for cluster {i}")


# In[160]:


plt.figure(figsize=(21,11))

#SSE
plt.subplot(2,3,1)
plt.plot([i for i in range(cluster_range[0],cluster_range[1])],km_sse,'b-o',linewidth=3,markersize=12)
plt.grid(True)
plt.title("SSE score vs. number of clusters",fontsize=15)
plt.xticks([i for i in range(0,cluster_range[1]+1,1)],fontsize=15)
plt.yticks(fontsize=15)

#Silhouette
plt.subplot(2,3,2)
plt.plot([i for i in range(cluster_range[0],cluster_range[1])],km_silhouette,'k-o',linewidth=3,markersize=12)
plt.grid(True)
plt.title("Silhouette score vs. number of clusters",fontsize=15)
plt.xticks([i for i in range(0,cluster_range[1]+1,1)],fontsize=15)
plt.yticks(fontsize=15)

#Homegeneity
plt.subplot(2,3,3)
plt.plot([i for i in range(cluster_range[0],cluster_range[1])],km_homogeneity,'r-o',linewidth=3,markersize=12)
plt.grid(True)
plt.title("Homegeneity score vs. number of clusters",fontsize=15)
plt.xticks([i for i in range(0,cluster_range[1]+1,1)],fontsize=15)
plt.yticks(fontsize=15)

#Completeness
plt.subplot(2,3,4)
plt.plot([i for i in range(cluster_range[0],cluster_range[1])],km_completeness,'g-o',linewidth=3,markersize=12)
plt.grid(True)
plt.title("Completeness score vs. number of clusters",fontsize=15)
plt.xticks([i for i in range(0,cluster_range[1]+1,1)],fontsize=15)
plt.yticks(fontsize=15)

#V-measure
plt.subplot(2,3,5)
plt.plot([i for i in range(cluster_range[0],cluster_range[1])],km_vmeasure,'c-o',linewidth=3,markersize=12)
plt.grid(True)
plt.title("V-measure score vs. number of clusters",fontsize=15)
plt.xticks([i for i in range(0,cluster_range[1]+1,1)],fontsize=15)
plt.yticks(fontsize=15)

#Adjusted mutual information
plt.subplot(2,3,6)
plt.plot([i for i in range(cluster_range[0],cluster_range[1])],km_ami,'m-o',linewidth=3,markersize=12)
plt.grid(True)
plt.title("Adjusted mutual information vs. number of clusters",fontsize=15)
plt.xticks([i for i in range(0,cluster_range[1]+1,1)],fontsize=15)
plt.yticks(fontsize=15)

plt.show()


# ### How the number of random projections affect clustering

# In[153]:


plt.figure(figsize=(20,6))
plt.title("SSE score vs. number of clusters for various random projections",fontsize=15)
for n in range(1,10,2):
    n_random_comp = n
    rp = GaussianRandomProjection(n_components=n_random_comp)
    X_rp = rp.fit_transform(X_scaled)
    #df_rp=pd.DataFrame(data=X_rp,columns=['Random_projection'+str(i) for i in range(1,n_random_comp+1)])
    km_sse= []
    km_silhouette = []
    cluster_range = (2,12)
    for i in range(cluster_range[0],cluster_range[1]):
        km = KMeans(n_clusters=i, random_state=0).fit(X_rp)
        preds = km.predict(X_rp)
        km_sse.append(-km.score(X_rp))
        km_silhouette.append(silhouette_score(X_rp,preds))
    # SSE plot
    plt.subplot(1,2,1)
    plt.plot([i for i in range(cluster_range[0],cluster_range[1])],km_sse,linewidth=3)
    plt.grid(True)    
    plt.xticks([i for i in range(0,cluster_range[1]+1,1)],fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlabel("Number of clusters", fontsize=15)
    plt.legend([str(n)+" random projections" for n in range(1,10,2)],fontsize=15)
    # Silhouette plot
    plt.subplot(1,2,2)
    plt.plot([i for i in range(cluster_range[0],cluster_range[1])],km_silhouette,linewidth=3)
    plt.grid(True)
    plt.xticks([i for i in range(0,cluster_range[1]+1,1)],fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlabel("Number of clusters", fontsize=18)
    plt.legend([str(n)+" random projections" for n in range(1,10,2)],fontsize=15)
plt.show()


# ### K-means fitting with random-projected data

# In[161]:


t1=time.time()
km_random_proj = KMeans(n_clusters=5,n_init=10,max_iter=500).fit(X=X_random_proj)
t2=time.time()
time_km_rp = t2-t1
preds_km_random_proj = km_random_proj.predict(X_random_proj)


# ### Visualizing the clusters after running k-means on random-projected features

# In[162]:


col_random_proj_combi=list(combinations(df_random_proj.columns,2))
num_random_proj_combi = len(col_random_proj_combi)


# In[164]:


plt.figure(figsize=(21,30))
for i in range(1,num_random_proj_combi+1):
    plt.subplot(int(num_random_proj_combi/3)+1,3,i)
    dim1=col_random_proj_combi[i-1][0]
    dim2=col_random_proj_combi[i-1][1]
    plt.scatter(df_random_proj[dim1],df_random_proj[dim2],c=preds_km_random_proj,edgecolor='k')
    plt.xlabel(f"{dim1}",fontsize=13)
    plt.ylabel(f"{dim2}",fontsize=13)
plt.show()


# ### Estimating average running time of k-means with random projections

# In[165]:


time_km_rp = []
for i in range(100):
    t1=time.time()
    km_rp = KMeans(n_clusters=5,n_init=10,max_iter=500).fit(X=X_random_proj)
    t2= time.time()
    delta = t2-t1
    time_km_rp.append(delta)
time_km_rp = np.array(time_km_rp)
avg_time_km_rp = time_km_rp.mean()
print(avg_time_km_rp)


# In[166]:


def plot_cluster_rp(df_rp,preds_rp):
    """
    Plots clusters after running random projection
    """
    plt.figure(figsize=(21,12))
    for i in range(1,num_random_proj_combi+1):
        plt.subplot(int(num_random_proj_combi/3)+1,3,i)
        dim1=col_random_proj_combi[i-1][0]
        dim2=col_random_proj_combi[i-1][1]
        plt.scatter(df_rp[dim1],df_rp[dim2],c=preds_rp,edgecolor='k')
        plt.xlabel(f"{dim1}",fontsize=13)
        plt.ylabel(f"{dim2}",fontsize=13)
    plt.show()


# ### Running the random projections many times

# In[228]:


rp_score= []
rp_silhouette = []
rp_vmeasure = []
n_random_comp = 7
for i in range(100):
    random_proj = GaussianRandomProjection(n_components=n_random_comp)
    X_random_proj = random_proj.fit_transform(X_scaled)
    df_random_proj=pd.DataFrame(data=X_random_proj,columns=['Random_projection'+str(i) for i in range(1,n_random_comp+1)])
    
    km = KMeans(n_clusters=5, random_state=0).fit(X_random_proj)
    preds = km.predict(X_random_proj)
    #print("Score for iteration {}: {}".format(i,km.score(X_random_proj)))
    rp_score.append(-km.score(X_random_proj))
    
    silhouette = silhouette_score(X_random_proj,preds)
    rp_silhouette.append(silhouette)
    #print("Silhouette score for iteration {}: {}".format(i,silhouette))
    
    v_measure = v_measure_score(y,preds)
    rp_vmeasure.append(v_measure)
    #print("V-measure score for iteration {}: {}".format(i,v_measure))
    #print("-"*100)
    
    #plot_cluster_rp(df_random_proj,preds)
    if (i%10==0):
        print("Done for {}".format(i))


# In[229]:


rp_score=np.array(rp_score)


# In[230]:


rp_score_scaled = rp_score/100


# In[231]:


rp_score_scaled


# In[232]:


rp_variations = {"SSE score":rp_score_scaled,"Silhouette score":rp_silhouette,"V measure":rp_vmeasure}


# In[233]:


df_rp_variations = pd.DataFrame(rp_variations)
df_rp_variations.head()


# In[252]:


df_rp_variations.plot.box(figsize=(8,5),patch_artist=True,color='orange')
plt.title("Variation in clustering metrics for\n 100 runs of random projections",fontsize=18)
plt.grid(True)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.ylim(0,1.1)
plt.show()


# ### This kind of variation does not happen with PCA

# In[93]:


pca_score= []
pca_silhouette = []
pca_vmeasure = []
for i in range(20):
    pca_partial = PCA(n_components=n_prin_comp,svd_solver='full')
    X_pca=pca_partial.fit_transform(X_scaled)
    km = KMeans(n_clusters=5, random_state=0).fit(X_pca)
    preds = km.predict(X_pca)
    print("Score for iteration {}: {}".format(i,km.score(X_pca)))
    rp_score.append(-km.score(X_pca))
    silhouette = silhouette_score(X_pca,preds)
    rp_silhouette.append(silhouette)
    print("Silhouette score for iteration {}: {}".format(i,silhouette))
    v_measure = v_measure_score(y,preds)
    rp_vmeasure.append(v_measure)
    print("V-measure score for iteration {}: {}".format(i,v_measure))
    print("-"*100)


# ### Testing cluster accuracy function

# In[239]:


def cluster_acc(Y,clusterLabels):
    import numpy as np
    from collections import Counter
    from sklearn.metrics import accuracy_score as acc
    assert (Y.shape == clusterLabels.shape)
    pred = np.empty_like(Y)
    for label in set(clusterLabels):
        mask = clusterLabels == label
        sub = Y[mask]
        target = Counter(sub).most_common(1)[0][0]
        pred[mask] = target
#    assert max(pred) == max(Y)
#    assert min(pred) == min(Y)    
    return acc(Y,pred)


# In[312]:


adjusted_rand_score(y,preds_gm)


# In[313]:


adjusted_rand_score(y,preds_km)


# In[314]:


adjusted_rand_score(y,preds_km_pca)


# In[315]:


adjusted_rand_score(y,preds_km_ica)


# In[316]:


adjusted_rand_score(y,preds_km_random_proj)


# ## Feature selection

# In[243]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel

print(X_scaled.shape)

clf = RandomForestClassifier(n_estimators=100,max_depth=3)
clf = clf.fit(X_scaled, y)
clf.feature_importances_  

model = SelectFromModel(clf, prefit=True)
X_fs = model.transform(X_scaled)

print(X_fs.shape)

fs_dim = X_fs.shape[1]


# In[245]:


df_fs = pd.DataFrame(X_fs,columns=['Selected_comp'+str(i) for i in range(1,fs_dim+1)])
df_fs.head()


# In[250]:


df_scaled = pd.DataFrame(X_scaled,columns=['Feature_'+str(i) for i in range(1,n_features+1)])
df_scaled.head()


# ### Running k-means on new transformed dataset after feature selection

# In[447]:


km_sse= []
km_silhouette = []
km_vmeasure =[]
km_ami = []
km_homogeneity = []
km_completeness = []

cluster_range = (2,12)

for i in range(cluster_range[0],cluster_range[1]):
    km = KMeans(n_clusters=i, random_state=0).fit(X_fs)
    preds = km.predict(X_fs)
    km_sse.append(-km.score(X_fs))
    km_silhouette.append(silhouette_score(X_fs,preds))
    km_vmeasure.append(v_measure_score(y,preds))
    km_ami.append(adjusted_mutual_info_score(y,preds))
    km_homogeneity.append(homogeneity_score(y,preds))
    km_completeness.append(completeness_score(y,preds))
    print(f"Done for cluster {i}")


# In[448]:


plt.figure(figsize=(21,11))

#SSE
plt.subplot(2,3,1)
plt.plot([i for i in range(cluster_range[0],cluster_range[1])],km_sse,'b-o',linewidth=3,markersize=12)
plt.grid(True)
plt.title("SSE score vs. number of clusters",fontsize=15)
plt.xticks([i for i in range(0,cluster_range[1]+1,1)],fontsize=15)
plt.yticks(fontsize=15)

#Silhouette
plt.subplot(2,3,2)
plt.plot([i for i in range(cluster_range[0],cluster_range[1])],km_silhouette,'k-o',linewidth=3,markersize=12)
plt.grid(True)
plt.title("Silhouette score vs. number of clusters",fontsize=15)
plt.xticks([i for i in range(0,cluster_range[1]+1,1)],fontsize=15)
plt.yticks(fontsize=15)

#Homegeneity
plt.subplot(2,3,3)
plt.plot([i for i in range(cluster_range[0],cluster_range[1])],km_homogeneity,'r-o',linewidth=3,markersize=12)
plt.grid(True)
plt.title("Homegeneity score vs. number of clusters",fontsize=15)
plt.xticks([i for i in range(0,cluster_range[1]+1,1)],fontsize=15)
plt.yticks(fontsize=15)

#Completeness
plt.subplot(2,3,4)
plt.plot([i for i in range(cluster_range[0],cluster_range[1])],km_completeness,'g-o',linewidth=3,markersize=12)
plt.grid(True)
plt.title("Completeness score vs. number of clusters",fontsize=15)
plt.xticks([i for i in range(0,cluster_range[1]+1,1)],fontsize=15)
plt.yticks(fontsize=15)

#V-measure
plt.subplot(2,3,5)
plt.plot([i for i in range(cluster_range[0],cluster_range[1])],km_vmeasure,'c-o',linewidth=3,markersize=12)
plt.grid(True)
plt.title("V-measure score vs. number of clusters",fontsize=15)
plt.xticks([i for i in range(0,cluster_range[1]+1,1)],fontsize=15)
plt.yticks(fontsize=15)

#Adjusted mutual information
plt.subplot(2,3,6)
plt.plot([i for i in range(cluster_range[0],cluster_range[1])],km_ami,'m-o',linewidth=3,markersize=12)
plt.grid(True)
plt.title("Adjusted mutual information vs. number of clusters",fontsize=15)
plt.xticks([i for i in range(0,cluster_range[1]+1,1)],fontsize=15)
plt.yticks(fontsize=15)

plt.show()


# ### Visualizing the clusters after feature selection

# In[246]:


t1=time.time()
km_fs = KMeans(n_clusters=5,n_init=10,max_iter=500).fit(X=X_fs)
t2=time.time()
time_km_fs = t2-t1

# Predictions
preds_km_fs = km_fs.predict(X_fs)


# In[247]:


col_fs_combi=list(combinations(df_fs.columns,2))
num_fs_combi = len(col_fs_combi)


# In[248]:


plt.figure(figsize=(21,20))
for i in range(1,num_fs_combi+1):
    plt.subplot(int(num_fs_combi/3)+1,3,i)
    dim1=col_fs_combi[i-1][0]
    dim2=col_fs_combi[i-1][1]
    plt.scatter(df_fs[dim1],df_fs[dim2],c=preds_km_fs,edgecolor='k')
    plt.xlabel(f"{dim1}",fontsize=13)
    plt.ylabel(f"{dim2}",fontsize=13)

