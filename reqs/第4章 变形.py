#!/usr/bin/env python
# coding: utf-8

# # 第4章 变形

# In[1]:


import numpy as np
import pandas as pd
df = pd.read_csv('data/table.csv')
df.head()


# ## 一、透视表
# ### 1. pivot
# #### 一般状态下，数据在DataFrame会以压缩（stacked）状态存放，例如上面的Gender，两个类别被叠在一列中，pivot函数可将某一列作为新的cols：

# In[2]:


df.pivot(index='ID',columns='Gender',values='Height').head()


# #### 然而pivot函数具有很强的局限性，除了功能上较少之外，还不允许values中出现重复的行列索引对（pair），例如下面的语句就会报错：

# In[3]:


#df.pivot(index='School',columns='Gender',values='Height').head()


# #### 因此，更多的时候会选择使用强大的pivot_table函数
# ### 2. pivot_table
# #### 首先，再现上面的操作：

# In[4]:


pd.pivot_table(df,index='ID',columns='Gender',values='Height').head()


# #### 由于功能更多，速度上自然是比不上原来的pivot函数：

# In[5]:


get_ipython().run_line_magic('timeit', "df.pivot(index='ID',columns='Gender',values='Height')")
get_ipython().run_line_magic('timeit', "pd.pivot_table(df,index='ID',columns='Gender',values='Height')")


# #### Pandas中提供了各种选项，下面介绍常用参数：
# #### ① aggfunc：对组内进行聚合统计，可传入各类函数，默认为'mean'

# In[6]:


pd.pivot_table(df,index='School',columns='Gender',values='Height',aggfunc=['mean','sum']).head()


# #### ② margins：汇总边际状态

# In[7]:


pd.pivot_table(df,index='School',columns='Gender',values='Height',aggfunc=['mean','sum'],margins=True).head()
#margins_name可以设置名字，默认为'All'


# #### ③ 行、列、值都可以为多级

# In[8]:


pd.pivot_table(df,index=['School','Class'],
               columns=['Gender','Address'],
               values=['Height','Weight'])


# ### 3. crosstab（交叉表）
# #### 交叉表是一种特殊的透视表，典型的用途如分组统计，如现在想要统计关于街道和性别分组的频数：

# In[9]:


pd.crosstab(index=df['Address'],columns=df['Gender'])


# #### 交叉表的功能也很强大（但目前还不支持多级分组），下面说明一些重要参数：
# #### ① values和aggfunc：分组对某些数据进行聚合操作，这两个参数必须成对出现

# In[10]:


pd.crosstab(index=df['Address'],columns=df['Gender'],
            values=np.random.randint(1,20,df.shape[0]),aggfunc='min')
#默认参数等于如下方法：
#pd.crosstab(index=df['Address'],columns=df['Gender'],values=1,aggfunc='count')


# #### ② 除了边际参数margins外，还引入了normalize参数，可选'all','index','columns'参数值

# In[11]:


pd.crosstab(index=df['Address'],columns=df['Gender'],normalize='all',margins=True)


# ## 二、其他变形方法
# ### 1. melt
# #### melt函数可以认为是pivot函数的逆操作，将unstacked状态的数据，压缩成stacked，使“宽”的DataFrame变“窄”

# In[12]:


df_m = df[['ID','Gender','Math']]
df_m.head()


# In[13]:


df.pivot(index='ID',columns='Gender',values='Math').head()


# #### melt函数中的id_vars表示需要保留的列，value_vars表示需要stack的一组列

# In[14]:


pivoted = df.pivot(index='ID',columns='Gender',values='Math')
result = pivoted.reset_index().melt(id_vars=['ID'],value_vars=['F','M'],value_name='Math')                     .dropna().set_index('ID').sort_index()
#检验是否与展开前的df相同，可以分别将这些链式方法的中间步骤展开，看看是什么结果
result.equals(df_m.set_index('ID'))


# ### 2. 压缩与展开
# #### （1）stack：这是最基础的变形函数，总共只有两个参数：level和dropna

# In[15]:


df_s = pd.pivot_table(df,index=['Class','ID'],columns='Gender',values=['Height','Weight'])
df_s.groupby('Class').head(2)


# In[16]:


df_stacked = df_s.stack()
df_stacked.groupby('Class').head(2)


# #### stack函数可以看做将横向的索引放到纵向，因此功能类似与melt，参数level可指定变化的列索引是哪一层（或哪几层，需要列表）

# In[17]:


df_stacked = df_s.stack(0)
df_stacked.groupby('Class').head(2)


# #### (2) unstack：stack的逆函数，功能上类似于pivot_table

# In[18]:


df_stacked.head()


# In[19]:


result = df_stacked.unstack().swaplevel(1,0,axis=1).sort_index(axis=1)
result.equals(df_s)
#同样在unstack中可以指定level参数


# ## 三、哑变量与因子化
# ### 1. Dummy Variable（哑变量）
# #### 这里主要介绍get_dummies函数，其功能主要是进行one-hot编码：

# In[20]:


df_d = df[['Class','Gender','Weight']]
df_d.head()


# #### 现在希望将上面的表格前两列转化为哑变量，并加入第三列Weight数值：

# In[21]:


pd.get_dummies(df_d[['Class','Gender']]).join(df_d['Weight']).head()
#可选prefix参数添加前缀，prefix_sep添加分隔符


# ### 2. factorize方法
# #### 该方法主要用于自然数编码，并且缺失值会被记做-1，其中sort参数表示是否排序后赋值

# In[22]:


codes, uniques = pd.factorize(['b', None, 'a', 'c', 'b'], sort=True)
display(codes)
display(uniques)


# ## 四、问题与练习

# ### 1. 问题
# #### 【问题一】 上面提到了许多变形函数，如melt/crosstab/pivot/pivot_table/stack/unstack函数，请总结它们各自的使用特点。

# #### 【问题二】 变形函数和多级索引是什么关系？哪些变形函数会使得索引维数变化？具体如何变化？
# #### 【问题三】 请举出一个除了上文提过的关于哑变量方法的例子。
# #### 【问题四】 使用完stack后立即使用unstack一定能保证变化结果与原始表完全一致吗？
# #### 【问题五】 透视表中涉及了三个函数，请分别使用它们完成相同的目标（任务自定）并比较哪个速度最快。
# #### 【问题六】 既然melt起到了stack的功能，为什么再设计stack函数？

# ### 2. 练习
# #### 【练习一】 继续使用上一章的药物数据集：

# In[23]:


pd.read_csv('data/Drugs.csv').head()


# #### (a) 现在请你将数据表转化成如下形态，每行需要显示每种药物在每个地区的10年至17年的变化情况，且前三列需要排序：
# ![avatar](picture/drug_pic.png)
# #### (b) 现在请将(a)中的结果恢复到原数据表，并通过equal函数检验初始表与新的结果是否一致（返回True）

# #### 【练习二】 现有一份关于某地区地震情况的数据集，请解决如下问题：

# In[24]:


pd.read_csv('data/Earthquake.csv').head()


# #### (a) 现在请你将数据表转化成如下形态，将方向列展开，并将距离、深度和烈度三个属性压缩：
# ![avatar](picture/earthquake_pic.png)
# #### (b) 现在请将(a)中的结果恢复到原数据表，并通过equal函数检验初始表与新的结果是否一致（返回True）
