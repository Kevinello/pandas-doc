#!/usr/bin/env python
# coding: utf-8

# # 第1章 Pandas基础

# In[ ]:


import pandas as pd
import numpy as np


# #### 查看Pandas版本

# In[ ]:


pd.__version__


# ## 一、文件读取与写入
# ### 1. 读取
# #### （a）csv格式

# In[3]:


df = pd.read_csv('data/table.csv')
df.head()


# #### （b）txt格式

# In[4]:


df_txt = pd.read_table('data/table.txt') #可设置sep分隔符参数
df_txt


# #### （c）xls或xlsx格式

# In[5]:


#需要安装xlrd包
df_excel = pd.read_excel('data/table.xlsx')
df_excel.head()


# ### 2. 写入

# #### （a）csv格式

# In[6]:


df.to_csv('data/new_table.csv')
#df.to_csv('data/new_table.csv', index=False) #保存时除去行索引


# #### （b）xls或xlsx格式

# In[7]:


#需要安装openpyxl
df.to_excel('data/new_table2.xlsx', sheet_name='Sheet1')


# ## 二、基本数据结构
# ### 1. Series
# #### （a）创建一个Series

# #### 对于一个Series，其中最常用的属性为值（values），索引（index），名字（name），类型（dtype）

# In[8]:


s = pd.Series(np.random.randn(5),index=['a','b','c','d','e'],name='这是一个Series',dtype='float64')
s


# #### （b）访问Series属性

# In[9]:


s.values


# In[10]:


s.name


# In[11]:


s.index


# In[12]:


s.dtype


# #### （c）取出某一个元素
# #### 将在第2章详细讨论索引的应用，这里先大致了解

# In[13]:


s['a']


# #### （d）调用方法

# In[14]:


s.mean()


# #### Series有相当多的方法可以调用：

# In[15]:


print([attr for attr in dir(s) if not attr.startswith('_')])


# ### 2. DataFrame
# #### （a）创建一个DataFrame

# In[16]:


df = pd.DataFrame({'col1':list('abcde'),'col2':range(5,10),'col3':[1.3,2.5,3.6,4.6,5.8]},
                 index=list('一二三四五'))
df


# #### （b）从DataFrame取出一列为Series

# In[17]:


df['col1']


# In[18]:


type(df)


# In[19]:


type(df['col1'])


# #### （c）修改行或列名

# In[20]:


df.rename(index={'一':'one'},columns={'col1':'new_col1'})


# #### （d）调用属性和方法

# In[21]:


df.index


# In[22]:


df.columns


# In[23]:


df.values


# In[24]:


df.shape


# In[25]:


df.mean() #本质上是一种Aggregation操作，将在第3章详细介绍


# #### （e）索引对齐特性
# #### 这是Pandas中非常强大的特性，不理解这一特性有时就会造成一些麻烦

# In[26]:


df1 = pd.DataFrame({'A':[1,2,3]},index=[1,2,3])
df2 = pd.DataFrame({'A':[1,2,3]},index=[3,1,2])
df1-df2 #由于索引对齐，因此结果不是0


# #### （f）列的删除与添加
# #### 对于删除而言，可以使用drop函数或del或pop

# In[27]:


df.drop(index='五',columns='col1') #设置inplace=True后会直接在原DataFrame中改动


# In[28]:


df['col1']=[1,2,3,4,5]
del df['col1']
df


# #### pop方法直接在原来的DataFrame上操作，且返回被删除的列，与python中的pop函数类似

# In[29]:


df['col1']=[1,2,3,4,5]
df.pop('col1')


# In[30]:


df


# #### 可以直接增加新的列，也可以使用assign方法

# In[31]:


df1['B']=list('abc')
df1


# In[32]:


df1.assign(C=pd.Series(list('def')))
#思考：为什么会出现NaN？（提示：索引对齐）assign左右两边的索引不一样，请问结果的索引谁说了算？


# #### 但assign方法不会对原DataFrame做修改

# In[33]:


df1


# #### （g）根据类型选择列

# In[34]:


df.select_dtypes(include=['number']).head()


# In[35]:


df.select_dtypes(include=['float']).head()


# #### （h）将Series转换为DataFrame

# In[36]:


s = df.mean()
s.name='to_DataFrame'
s


# In[37]:


s.to_frame()


# #### 使用T符号可以转置

# In[38]:


s.to_frame().T


# ## 三、常用基本函数
# #### 从下面开始，包括后面所有章节，我们都会用到这份虚拟的数据集

# In[39]:


df = pd.read_csv('data/table.csv')


# ### 1. head和tail

# In[40]:


df.head()


# In[41]:


df.tail()


# #### 可以指定n参数显示多少行

# In[42]:


df.head(3)


# ### 2. unique和nunique

# #### nunique显示有多少个唯一值

# In[43]:


df['Physics'].nunique()


# #### unique显示所有的唯一值

# In[44]:


df['Physics'].unique()


# ### 3. count和value_counts

# #### count返回非缺失值元素个数

# In[45]:


df['Physics'].count()


# #### value_counts返回每个元素有多少个

# In[46]:


df['Physics'].value_counts()


# ### 4. describe和info

# #### info函数返回有哪些列、有多少非缺失值、每列的类型

# In[47]:


df.info()


# #### describe默认统计数值型数据的各个统计量

# In[48]:


df.describe()


# #### 可以自行选择分位数

# In[49]:


df.describe(percentiles=[.05, .25, .75, .95])


# #### 对于非数值型也可以用describe函数

# In[50]:


df['Physics'].describe()


# ### 5. idxmax和nlargest
# #### idxmax函数返回最大值所在索引，在某些情况下特别适用，idxmin功能类似

# In[51]:


df['Math'].idxmax()


# #### nlargest函数返回前几个大的元素值，nsmallest功能类似

# In[52]:


df['Math'].nlargest(3)


# ### 6. clip和replace

# #### clip和replace是两类替换函数
# #### clip是对超过或者低于某些值的数进行截断

# In[53]:


df['Math'].head()


# In[54]:


df['Math'].clip(33,80).head()


# In[55]:


df['Math'].mad()


# #### replace是对某些值进行替换

# In[56]:


df['Address'].head()


# In[57]:


df['Address'].replace(['street_1','street_2'],['one','two']).head()


# #### 通过字典，可以直接在表中修改

# In[58]:


df.replace({'Address':{'street_1':'one','street_2':'two'}}).head()


# ### 7. apply函数
# #### apply是一个自由度很高的函数，在第3章我们还要提到
# #### 对于Series，它可以迭代每一列的值操作：

# In[59]:


df['Math'].apply(lambda x:str(x)+'!').head() #可以使用lambda表达式，也可以使用函数


# #### 对于DataFrame，它在默认axis=0下可以迭代每一个列操作：

# In[60]:


df.apply(lambda x:x.apply(lambda x:str(x)+'!')).head() #这是一个稍显复杂的例子，有利于理解apply的功能


# #### Pandas中的axis参数=0时，永远表示的是处理方向而不是聚合方向，当axis='index'或=0时，对列迭代对行聚合，行即为跨列，axis=1同理

# ## 四、排序

# ### 1. 索引排序

# In[61]:


df.set_index('Math').head() #set_index函数可以设置索引，将在下一章详细介绍


# In[62]:


df.set_index('Math').sort_index().head() #可以设置ascending参数，默认为升序，True


# ### 2. 值排序

# In[63]:


df.sort_values(by='Class').head()


# #### 多个值排序，即先对第一层排，在第一层相同的情况下对第二层排序

# In[64]:


df.sort_values(by=['Address','Height']).head()


# ## 五、问题与练习
# ### 1. 问题
# #### 【问题一】 Series和DataFrame有哪些常见属性和方法？
# #### 【问题二】 value_counts会统计缺失值吗？
# #### 【问题三】 如果有多个索引同时取到最大值，idxmax会返回所有这些索引吗？如果不会，那么怎么返回这些索引？
# #### 【问题四】 在常用函数一节中，由于一些函数的功能比较简单，因此没有列入，现在将它们列在下面，请分别说明它们的用途并尝试使用。
# #### sum/mean/median/mad/min/max/abs/std/var/quantile/cummax/cumsum/cumprod
# #### 【问题五】 df.mean(axis=1)是什么意思？它与df.mean()的结果一样吗？问题四提到的函数也有axis参数吗？怎么使用？
# #### 【问题六】 对值进行排序后，相同的值次序由什么决定？
# #### 【问题七】 Pandas中为各类基础运算也定义了函数，比如s1.add(s2)表示两个Series相加，但既然已经有了'+'，是不是多此一举？

# ### 2. 练习
# #### 【练习一】 现有一份关于美剧《权力的游戏》剧本的数据集，请解决以下问题：
# #### （a）在所有的数据中，一共出现了多少人物？
# #### （b）以单元格计数（即简单把一个单元格视作一句），谁说了最多的话？
# #### （c）以单词计数，谁说了最多的单词？（不是单句单词最多，是指每人说过单词的总数最多，为了简便，只以空格为单词分界点，不考虑其他情况）

# In[65]:


pd.read_csv('data/Game_of_Thrones_Script.csv').head()


# #### 【练习二】现有一份关于科比的投篮数据集，请解决如下问题：
# #### （a）哪种action_type和combined_shot_type的组合是最多的？
# #### （b）在所有被记录的game_id中，遭遇到最多的opponent是一个支？（由于一场比赛会有许多次投篮，但对阵的对手只有一个，本题相当于问科比和哪个队交锋次数最多）

# In[66]:


pd.read_csv('data/Kobe_data.csv',index_col='shot_id').head()
#index_col的作用是将某一列作为行索引

