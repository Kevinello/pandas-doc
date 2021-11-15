#!/usr/bin/env python
# coding: utf-8

# # 第10章 不定期更新的例子

# In[1]:


import pandas as pd
import numpy as np
import time


# ## 一、评委打分
# #### 某比赛有1000名选手，300位评委打分，每个选手由三个不同的评委打分，每位评委打10位选手的分
# #### 现在需要将各个评委的编号转到列索引，行索引不变，表格内容为打分分数，缺失值（即选手i没有被评委j打分）用'-'填充

# In[2]:


df = pd.read_csv('data/Competition.csv',index_col='选手编号')
df.head()


# #### 【方法一】思维量较大，有技巧性，对Pandas依赖较少

# In[3]:


t0=time.perf_counter()
############################################################################################
L,k = [],1
for i in range(301):
    judge = 'Judge_%d'%i
    result = df[(df.iloc[:,0::2]==judge).any(1)]
    L_temp = (result.iloc[:,0::2]==judge).values*result.iloc[:,1::2].values
    L.append(list(zip(result.index.tolist(),list(L_temp.max(axis=1)))))
L.pop(0)
df_result = pd.DataFrame([['-']*1000]*300,index=['Judge_%d'%i for i in range(1,301)]
                         ,columns=['%d'%i for i in range(1,1001)])
for i in L:
    for j in i:
        df_result.at['Judge_%d'%k,'%d'%j[0]] = j[1]
    k += 1
############################################################################################
t1=time.perf_counter()
print('时间为：%.3f'%(t1-t0))
df_result.T.head()


# #### 【方法二】思路简单，但运行时间较长

# In[4]:


t0=time.perf_counter()
############################################################################################
judge = np.array([[df.iloc[:,0:2].values],[df.iloc[:,2:4].values],[df.iloc[:,4:6].values]]).reshape(6000)[0::2]
score = np.array([[df.iloc[:,0:2].values],[df.iloc[:,2:4].values],[df.iloc[:,4:6].values]]).reshape(6000)[1::2]
df_result = pd.DataFrame({'judge':judge,'score':score}
                         ,index=np.array([range(1,1001)]*3).reshape(3000)).reset_index()
df_result = pd.crosstab(index=df_result['index'],columns=df_result['judge'],values=df_result['score']
                     ,aggfunc=np.sum).fillna('-').T.reindex(['Judge_%d'%i for i in range(1,301)]).T
############################################################################################
t1=time.perf_counter()
print('时间为：%.3f'%(t1-t0))
df_result.head()


# #### 【方法三】基本与方法二类似，但借助pivot函数大幅提高速度

# In[5]:


t0=time.perf_counter()
############################################################################################
judge = np.array([[df.iloc[:,0:2].values],[df.iloc[:,2:4].values],[df.iloc[:,4:6].values]]).reshape(6000)[0::2]
score = np.array([[df.iloc[:,0:2].values],[df.iloc[:,2:4].values],[df.iloc[:,4:6].values]]).reshape(6000)[1::2]
df_result = pd.DataFrame({'judge':judge,'score':score}
                         ,index=np.array([range(1,1001)]*3).reshape(3000)).reset_index()
df_result = df_result.pivot(index='index',columns='judge'
                    ,values='score').T.reindex(['Judge_%d'%i for i in range(1,301)]).T.fillna('-')
############################################################################################
t1=time.perf_counter()
print('时间为：%.3f'%(t1-t0))
df_result.head()


# ## 二、企业收入熵指数
# #### 一个企业的产业多元化水平可以由收入熵指数计算衡量，其公式为$-\Sigma P_i \ln{P_i}$，其中i表示第i个收入类型，$P_i$表示该类型收入额所占整个收入额的比重（因此$\Sigma P_i=1$），现在需要对Company.csv中的公司计算它们的年度收入熵，需要利用Company_data.csv中不同收入类型销售额的数据（证券代码都是六位，第一列数字需要补零），请计算结果并保存到data文件夹下
# #### 注意：不是所有要求计算的公司都会在data文件中出现，反之亦然；某公司某年的数据若含有缺失值，请基于收入熵公式选择一种合理的计算方式

# In[6]:


df_c = pd.read_csv('data/Company.csv')
df_c.head()


# In[7]:


df = pd.read_csv('data/Company_data.csv')
df.head()


# #### 【参考答案】

# In[8]:


df_c = pd.read_csv('data/Company.csv')
df = pd.read_csv('data/Company_data.csv')
df['证券代码'] = df['证券代码'].apply(lambda x:'#'+'0'*(6-len(str(x)))+str(x))
df['日期'] = pd.to_datetime(df['日期']).dt.year
df_new = df[df['证券代码'].apply(lambda x:True if x in df_c['证券代码'].values else False)]
result = pd.merge(df_c, df_new.groupby(['证券代码','日期'])['收入额'].agg(lambda x:sum([
    -i*np.log(i) for i in x[x>0]/sum(x[x>0])])).reset_index(), on=['证券代码','日期'], how='left')


# In[9]:


result.rename(columns={'收入额':'收入熵'})

