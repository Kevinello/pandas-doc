#!/usr/bin/env python
# coding: utf-8

# # 第5章 合并

# In[1]:


import numpy as np
import pandas as pd
df = pd.read_csv('data/table.csv')
df.head()


# ## 一、append与assign
# ### 1. append方法
# #### （a）利用序列添加行（必须指定name）

# In[2]:


df_append = df.loc[:3,['Gender','Height']].copy()
df_append


# In[3]:


s = pd.Series({'Gender':'F','Height':188},name='new_row')
df_append.append(s)


# #### （b）用DataFrame添加表

# In[4]:


df_temp = pd.DataFrame({'Gender':['F','M'],'Height':[188,176]},index=['new_1','new_2'])
df_append.append(df_temp)


# ### 2. assign方法
# #### 该方法主要用于添加列，列名直接由参数指定：

# In[5]:


s = pd.Series(list('abcd'),index=range(4))
df_append.assign(Letter=s)


# #### 可以一次添加多个列：

# In[6]:


df_append.assign(col1=lambda x:x['Gender']*2,
                 col2=s)


# ## 二、combine与update
# ### 1. comine方法
# #### comine和update都是用于表的填充函数，可以根据某种规则填充
# #### （a）填充对象
# #### 可以看出combine方法是按照表的顺序轮流进行逐列循环的，而且自动索引对齐，缺失值为NaN，理解这一点很重要

# In[7]:


df_combine_1 = df.loc[:1,['Gender','Height']].copy()
df_combine_2 = df.loc[10:11,['Gender','Height']].copy()
df_combine_1.combine(df_combine_2,lambda x,y:print(x,y))


# #### （b）一些例子
# #### 例①：根据列均值的大小填充

# In[8]:


# 例子1
df1 = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
df2 = pd.DataFrame({'A': [8, 7], 'B': [6, 5]})
df1.combine(df2,lambda x,y:x if x.mean()>y.mean() else y)


# #### 例②：索引对齐特性（默认状态下，后面的表没有的行列都会设置为NaN）

# In[9]:


df2 = pd.DataFrame({'B': [8, 7], 'C': [6, 5]},index=[1,2])
df1.combine(df2,lambda x,y:x if x.mean()>y.mean() else y)


# #### 例③：使得df1原来符合条件的值不会被覆盖

# In[10]:


df1.combine(df2,lambda x,y:x if x.mean()>y.mean() else y,overwrite=False) 


# #### 例④：在新增匹配df2的元素位置填充-1

# In[11]:


df1.combine(df2,lambda x,y:x if x.mean()>y.mean() else y,fill_value=-1)


# #### （c）combine_first方法
# #### 这个方法作用是用df2填补df1的缺失值，功能比较简单，但很多时候会比combine更常用，下面举两个例子：

# In[12]:


df1 = pd.DataFrame({'A': [None, 0], 'B': [None, 4]})
df2 = pd.DataFrame({'A': [1, 1], 'B': [3, 3]})
df1.combine_first(df2)


# In[13]:


df1 = pd.DataFrame({'A': [None, 0], 'B': [4, None]})
df2 = pd.DataFrame({'B': [3, 3], 'C': [1, 1]}, index=[1, 2])
df1.combine_first(df2)


# ### 2. update方法
# #### （a）三个特点
# #### ①返回的框索引只会与被调用框的一致（默认使用左连接，下一节会介绍）
# #### ②第二个框中的nan元素不会起作用
# #### ③没有返回值，直接在df上操作
# #### （b）例子
# #### 例①：索引完全对齐情况下的操作

# In[14]:


df1 = pd.DataFrame({'A': [1, 2, 3],
                    'B': [400, 500, 600]})
df2 = pd.DataFrame({'B': [4, 5, 6],
                    'C': [7, 8, 9]})
df1.update(df2)
df1


# #### 例②：部分填充

# In[15]:


df1 = pd.DataFrame({'A': ['a', 'b', 'c'],
                    'B': ['x', 'y', 'z']})
df2 = pd.DataFrame({'B': ['d', 'e']}, index=[1,2])
df1.update(df2)
df1


# #### 例③：缺失值不会填充

# In[16]:


df1 = pd.DataFrame({'A': [1, 2, 3],
                    'B': [400, 500, 600]})
df2 = pd.DataFrame({'B': [4, np.nan, 6]})
df1.update(df2)
df1


# ## 三、concat方法
# #### concat方法可以在两个维度上拼接，默认纵向凭借（axis=0），拼接方式默认外连接
# #### 所谓外连接，就是取拼接方向的并集，而'inner'时取拼接方向（若使用默认的纵向拼接，则为列的交集）的交集
# #### 下面举一些例子说明其参数：

# In[17]:


df1 = pd.DataFrame({'A': ['A0', 'A1'],
                    'B': ['B0', 'B1']},
                    index = [0,1])
df2 = pd.DataFrame({'A': ['A2', 'A3'],
                    'B': ['B2', 'B3']},
                    index = [2,3])
df3 = pd.DataFrame({'A': ['A1', 'A3'],
                    'D': ['D1', 'D3'],
                    'E': ['E1', 'E3']},
                    index = [1,3])


# #### 默认状态拼接：

# In[18]:


pd.concat([df1,df2])


# #### axis=1时沿列方向拼接：

# In[19]:


pd.concat([df1,df2],axis=1)


# #### join设置为内连接（由于axis=0，因此列取交集）：

# In[20]:


pd.concat([df3,df1],join='inner')


# #### join设置为外链接：

# In[21]:


pd.concat([df3,df1],join='outer',sort=True) #sort设置列排序，默认为False


# #### verify_integrity检查列是否唯一：

# In[22]:


#pd.concat([df3,df1],verify_integrity=True,sort=True) 报错


# #### 同样，可以添加Series：

# In[23]:


s = pd.Series(['X0', 'X1'], name='X')
pd.concat([df1,s],axis=1)


# #### key参数用于对不同的数据框增加一个标号，便于索引：

# In[24]:


pd.concat([df1,df2], keys=['x', 'y'])
pd.concat([df1,df2], keys=['x', 'y']).index


# ## 四、merge与join
# ### 1. merge函数
# #### merge函数的作用是将两个pandas对象横向合并，遇到重复的索引项时会使用笛卡尔积，默认inner连接，可选left、outer、right连接
# #### 所谓左连接，就是指以第一个表索引为基准，右边的表中如果不再左边的则不加入，如果在左边的就以笛卡尔积的方式加入
# #### merge/join与concat的不同之处在于on参数，可以指定某一个对象为key来进行连接
# #### 同样的，下面举一些例子：

# In[25]:


left = pd.DataFrame({'key1': ['K0', 'K0', 'K1', 'K2'],
                     'key2': ['K0', 'K1', 'K0', 'K1'],
                      'A': ['A0', 'A1', 'A2', 'A3'],
                      'B': ['B0', 'B1', 'B2', 'B3']}) 
right = pd.DataFrame({'key1': ['K0', 'K1', 'K1', 'K2'],
                      'key2': ['K0', 'K0', 'K0', 'K0'],
                      'C': ['C0', 'C1', 'C2', 'C3'],
                      'D': ['D0', 'D1', 'D2', 'D3']})
right2 = pd.DataFrame({'key1': ['K0', 'K1', 'K1', 'K2'],
                      'key2': ['K0', 'K0', 'K0', 'K0'],
                      'C': ['C0', 'C1', 'C2', 'C3']})


# #### 以key1为准则连接，如果具有相同的列，则默认suffixes=('_x','_y')：

# In[26]:


pd.merge(left, right, on='key1')


# #### 以多组键连接：

# In[27]:


pd.merge(left, right, on=['key1','key2'])


# #### 默认使用inner连接，因为merge只能横向拼接，所以取行向上keys的交集，下面看如果使用how=outer参数
# #### 注意：这里的how就是concat的join

# In[28]:


pd.merge(left, right, how='outer', on=['key1','key2'])


# #### 左连接：

# In[29]:


pd.merge(left, right, how='left', on=['key1', 'key2'])


# #### 右连接：

# In[30]:


pd.merge(left, right, how='right', on=['key1', 'key2'])


# #### 如果还是对笛卡尔积不太了解，请务必理解下面这个例子，由于B的所有元素为2，因此需要6行：

# In[31]:


left = pd.DataFrame({'A': [1, 2], 'B': [2, 2]})
right = pd.DataFrame({'A': [4, 5, 6], 'B': [2, 2, 2]})
pd.merge(left, right, on='B', how='outer')


# #### validate检验的是到底哪一边出现了重复索引，如果是“one_to_one”则两侧索引都是唯一，如果"one_to_many"则左侧唯一

# In[32]:


left = pd.DataFrame({'A': [1, 2], 'B': [2, 2]})
right = pd.DataFrame({'A': [4, 5, 6], 'B': [2, 3, 4]})
#pd.merge(left, right, on='B', how='outer',validate='one_to_one') #报错


# In[33]:


left = pd.DataFrame({'A': [1, 2], 'B': [2, 1]})
pd.merge(left, right, on='B', how='outer',validate='one_to_one')


# #### indicator参数指示了，合并后该行索引的来源

# In[34]:


df1 = pd.DataFrame({'col1': [0, 1], 'col_left': ['a', 'b']})
df2 = pd.DataFrame({'col1': [1, 2, 2], 'col_right': [2, 2, 2]})
pd.merge(df1, df2, on='col1', how='outer', indicator=True) #indicator='indicator_column'也是可以的


# ### 2. join函数
# #### join函数作用是将多个pandas对象横向拼接，遇到重复的索引项时会使用笛卡尔积，默认左连接，可选inner、outer、right连接

# In[35]:


left = pd.DataFrame({'A': ['A0', 'A1', 'A2'],
                     'B': ['B0', 'B1', 'B2']},
                    index=['K0', 'K1', 'K2'])
right = pd.DataFrame({'C': ['C0', 'C2', 'C3'],
                      'D': ['D0', 'D2', 'D3']},
                    index=['K0', 'K2', 'K3'])
left.join(right)


# #### 对于many_to_one模式下的合并，往往join更为方便
# #### 同样可以指定key：

# In[36]:


left = pd.DataFrame({'A': ['A0', 'A1', 'A2', 'A3'],
                     'B': ['B0', 'B1', 'B2', 'B3'],
                     'key': ['K0', 'K1', 'K0', 'K1']})
right = pd.DataFrame({'C': ['C0', 'C1'],
                      'D': ['D0', 'D1']},
                     index=['K0', 'K1'])
left.join(right, on='key')


# #### 多层key：

# In[37]:


left = pd.DataFrame({'A': ['A0', 'A1', 'A2', 'A3'],
                     'B': ['B0', 'B1', 'B2', 'B3'],
                     'key1': ['K0', 'K0', 'K1', 'K2'],
                     'key2': ['K0', 'K1', 'K0', 'K1']})
index = pd.MultiIndex.from_tuples([('K0', 'K0'), ('K1', 'K0'),
                                   ('K2', 'K0'), ('K2', 'K1')],names=['key1','key2'])
right = pd.DataFrame({'C': ['C0', 'C1', 'C2', 'C3'],
                      'D': ['D0', 'D1', 'D2', 'D3']},
                     index=index)
left.join(right, on=['key1','key2'])


# ## 五、问题与练习

# ### 1. 问题
# #### 【问题一】 请思考什么是append/assign/combine/update/concat/merge/join各自最适合使用的场景，并举出相应的例子。
# #### 【问题二】 merge_ordered和merge_asof的作用是什么？和merge是什么关系？
# #### 【问题三】 请构造一个多级索引与多级索引合并的例子，尝试使用不同的合并函数。
# #### 【问题四】 上文提到了连接的笛卡尔积，那么当连接方式变化时（inner/outer/left/right），这种笛卡尔积规则会相应变化吗？请构造相应例子。

# ### 2. 练习
# #### 【练习一】有2张公司的员工信息表，每个公司共有16名员工，共有五个公司，请解决如下问题：

# In[38]:


pd.read_csv('data/Employee1.csv').head()


# In[39]:


pd.read_csv('data/Employee2.csv').head()


# #### (a) 每个公司有多少员工满足如下条件：既出现第一张表，又出现在第二张表。
# #### (b) 将所有不符合(a)中条件的行筛选出来，合并为一张新表，列名与原表一致。
# #### (c) 现在需要编制所有80位员工的信息表，对于(b)中的员工要求不变，对于满足(a)条件员工，它们在某个指标的数值，取偏离它所属公司中满足(b)员工的均值数较小的哪一个，例如：P公司在两张表的交集为{p1}，并集扣除交集为{p2,p3,p4}，那么如果后者集合的工资均值为1万元，且p1在表1的工资为13000元，在表2的工资为9000元，那么应该最后取9000元作为p1的工资，最后对于没有信息的员工，利用缺失值填充。

# #### 【练习二】有2张课程的分数表（分数随机生成），但专业课（学科基础课、专业必修课、专业选修课）与其他课程混在一起，请解决如下问题：

# In[40]:


pd.read_csv('data/Course1.csv').head()


# In[41]:


pd.read_csv('data/Course2.csv').head()


# #### (a) 将两张表分别拆分为专业课与非专业课（结果为四张表）。
# #### (b) 将两张专业课的分数表和两张非专业课的分数表分别合并。
# #### (c) 不使用(a)中的步骤，请直接读取两张表合并后拆分。
# #### (d) 专业课程中有缺失值吗，如果有的话请在完成(3)的同时，用组内（3种类型的专业课）均值填充缺失值后拆分。
