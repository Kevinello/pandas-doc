#!/usr/bin/env python
# coding: utf-8

# # 第6章 缺失数据

# #### 在接下来的两章中，会接触到数据预处理中比较麻烦的类型，即缺失数据和文本数据（尤其是混杂型文本）
# #### Pandas在步入1.0后，对数据类型也做出了新的尝试，尤其是Nullable类型和String类型，了解这些可能在未来成为主流的新特性是必要的

# In[1]:


import pandas as pd
import numpy as np
df = pd.read_csv('data/table_missing.csv')
df.head()


# ## 一、缺失观测及其类型

# ### 1. 了解缺失信息
# #### （a）isna和notna方法
# #### 对Series使用会返回布尔列表

# In[2]:


df['Physics'].isna().head()


# In[3]:


df['Physics'].notna().head()


# #### 对DataFrame使用会返回布尔表

# In[4]:


df.isna().head()


# #### 但对于DataFrame我们更关心到底每列有多少缺失值

# In[5]:


df.isna().sum()


# #### 此外，可以通过第1章中介绍的info函数查看缺失信息

# In[6]:


df.info()


# #### （b）查看缺失值的所以在行

# #### 以最后一列为例，挑出该列缺失值的行

# In[7]:


df[df['Physics'].isna()]


# #### （c）挑选出所有非缺失值列
# #### 使用all就是全部非缺失值，如果是any就是至少有一个不是缺失值

# In[8]:


df[df.notna().all(1)]


# ### 2. 三种缺失符号
# #### （a）np.nan
# #### np.nan是一个麻烦的东西，首先它不等与任何东西，甚至不等于自己

# In[9]:


np.nan == np.nan


# In[10]:


np.nan == 0


# In[11]:


np.nan == None


# #### 在用equals函数比较时，自动略过两侧全是np.nan的单元格，因此结果不会影响

# In[12]:


df.equals(df)


# #### 其次，它在numpy中的类型为浮点，由此导致数据集读入时，即使原来是整数的列，只要有缺失值就会变为浮点型

# In[13]:


type(np.nan)


# In[14]:


pd.Series([1,2,3]).dtype


# In[15]:


pd.Series([1,np.nan,3]).dtype


# #### 此外，对于布尔类型的列表，如果是np.nan填充，那么它的值会自动变为True而不是False

# In[16]:


pd.Series([1,np.nan,3],dtype='bool')


# #### 但当修改一个布尔列表时，会改变列表类型，而不是赋值为True

# In[17]:


s = pd.Series([True,False],dtype='bool')
s[1]=np.nan
s


# #### 在所有的表格读取后，无论列是存放什么类型的数据，默认的缺失值全为np.nan类型
# #### 因此整型列转为浮点；而字符由于无法转化为浮点，因此只能归并为object类型（'O'），原来是浮点型的则类型不变

# In[18]:


df['ID'].dtype


# In[19]:


df['Math'].dtype


# In[20]:


df['Class'].dtype


# #### （b）None
# #### None比前者稍微好些，至少它会等于自身

# In[21]:


None == None


# #### 它的布尔值为False

# In[22]:


pd.Series([None],dtype='bool')


# #### 修改布尔列表不会改变数据类型

# In[23]:


s = pd.Series([True,False],dtype='bool')
s[0]=None
s


# In[24]:


s = pd.Series([1,0],dtype='bool')
s[0]=None
s


# #### 在传入数值类型后，会自动变为np.nan

# In[25]:


type(pd.Series([1,None])[1])


# #### 只有当传入object类型是保持不动，几乎可以认为，除非人工命名None，它基本不会自动出现在Pandas中

# In[26]:


type(pd.Series([1,None],dtype='O')[1])


# ####  在使用equals函数时不会被略过，因此下面的情况下返回False

# In[27]:


pd.Series([None]).equals(pd.Series([np.nan]))


# #### （c）NaT
# #### NaT是针对时间序列的缺失值，是Pandas的内置类型，可以完全看做时序版本的np.nan，与自己不等，且使用equals是也会被跳过

# In[28]:


s_time = pd.Series([pd.Timestamp('20120101')]*5)
s_time


# In[29]:


s_time[2] = None
s_time


# In[30]:


s_time[2] = np.nan
s_time


# In[31]:


s_time[2] = pd.NaT
s_time


# In[32]:


type(s_time[2])


# In[33]:


s_time[2] == s_time[2]


# In[34]:


s_time.equals(s_time)


# In[35]:


s = pd.Series([True,False],dtype='bool')
s[1]=pd.NaT
s


# ### 3. Nullable类型与NA符号
# #### 这是Pandas在1.0新版本中引入的重大改变，其目的就是为了（在若干版本后）解决之前出现的混乱局面，统一缺失值处理方法
# #### "The goal of pd.NA is provide a “missing” indicator that can be used consistently across data types (instead of np.nan, None or pd.NaT depending on the data type)."——User Guide for Pandas v-1.0
# #### 官方鼓励用户使用新的数据类型和缺失类型pd.NA

# #### （a）Nullable整形
# #### 对于该种类型而言，它与原来标记int上的符号区别在于首字母大写：'Int'

# In[36]:


s_original = pd.Series([1, 2], dtype="int64")
s_original


# In[37]:


s_new = pd.Series([1, 2], dtype="Int64")
s_new


# #### 它的好处就在于，其中前面提到的三种缺失值都会被替换为统一的NA符号，且不改变数据类型

# In[38]:


s_original[1] = np.nan
s_original


# In[39]:


s_new[1] = np.nan
s_new


# In[40]:


s_new[1] = None
s_new


# In[41]:


s_new[1] = pd.NaT
s_new


# #### （b）Nullable布尔
# #### 对于该种类型而言，作用与上面的类似，记号为boolean

# In[42]:


s_original = pd.Series([1, 0], dtype="bool")
s_original


# In[43]:


s_new = pd.Series([0, 1], dtype="boolean")
s_new


# In[44]:


s_original[0] = np.nan
s_original


# In[45]:


s_original = pd.Series([1, 0], dtype="bool") #此处重新加一句是因为前面赋值改变了bool类型
s_original[0] = None
s_original


# In[46]:


s_new[0] = np.nan
s_new


# In[47]:


s_new[0] = None
s_new


# In[48]:


s_new[0] = pd.NaT
s_new


# #### 需要注意的是，含有pd.NA的布尔列表在1.0.2之前的版本作为索引时会报错，这是一个之前的[bug](https://pandas.pydata.org/docs/whatsnew/v1.0.2.html#indexing-with-nullable-boolean-arrays)，现已经修复

# In[49]:


s = pd.Series(['dog','cat'])
s[s_new]


# #### （c）string类型
# #### 该类型是1.0的一大创新，目的之一就是为了区分开原本含糊不清的object类型，这里将简要地提及string，因为它是第7章的主题内容
# #### 它本质上也属于Nullable类型，因为并不会因为含有缺失而改变类型

# In[50]:


s = pd.Series(['dog','cat'],dtype='string')
s


# In[51]:


s[0] = np.nan
s


# In[52]:


s[0] = None
s


# In[53]:


s[0] = pd.NaT
s


# #### 此外，和object类型的一点重要区别就在于，在调用字符方法后，string类型返回的是Nullable类型，object则会根据缺失类型和数据类型而改变

# In[54]:


s = pd.Series(["a", None, "b"], dtype="string")
s.str.count('a')


# In[55]:


s2 = pd.Series(["a", None, "b"], dtype="object")
s2.str.count("a")


# In[56]:


s.str.isdigit()


# In[57]:


s2.str.isdigit()


# ### 4. NA的特性

# #### （a）逻辑运算
# #### 只需看该逻辑运算的结果是否依赖pd.NA的取值，如果依赖，则结果还是NA，如果不依赖，则直接计算结果

# In[58]:


True | pd.NA


# In[59]:


pd.NA | True


# In[60]:


False | pd.NA


# In[61]:


False & pd.NA


# In[62]:


True & pd.NA


# #### 取值不明直接报错

# In[63]:


#bool(pd.NA)


# #### （b）算术运算和比较运算
# #### 这里只需记住除了下面两类情况，其他结果都是NA即可

# In[64]:


pd.NA ** 0


# In[65]:


1 ** pd.NA


# #### 其他情况：

# In[66]:


pd.NA + 1


# In[67]:


"a" * pd.NA


# In[68]:


pd.NA == pd.NA


# In[69]:


pd.NA < 2.5


# In[70]:


np.log(pd.NA)


# In[71]:


np.add(pd.NA, 1)


# ### 5.  convert_dtypes方法
# #### 这个函数的功能往往就是在读取数据时，就把数据列转为Nullable类型，是1.0的新函数

# In[72]:


pd.read_csv('data/table_missing.csv').dtypes


# In[73]:


pd.read_csv('data/table_missing.csv').convert_dtypes().dtypes


# ## 二、缺失数据的运算与分组

# ### 1. 加号与乘号规则

# #### 使用加法时，缺失值为0

# In[74]:


s = pd.Series([2,3,np.nan,4])
s.sum()


# #### 使用乘法时，缺失值为1

# In[75]:


s.prod()


# #### 使用累计函数时，缺失值自动略过

# In[76]:


s.cumsum()


# In[77]:


s.cumprod()


# In[78]:


s.pct_change()


# ### 2. groupby方法中的缺失值
# #### 自动忽略为缺失值的组

# In[79]:


df_g = pd.DataFrame({'one':['A','B','C','D',np.nan],'two':np.random.randn(5)})
df_g


# In[80]:


df_g.groupby('one').groups


# ## 三、填充与剔除

# ### 1. fillna方法

# #### （a）值填充与前后向填充（分别与ffill方法和bfill方法等价）

# In[81]:


df['Physics'].fillna('missing').head()


# In[82]:


df['Physics'].fillna(method='ffill').head()


# In[83]:


df['Physics'].fillna(method='backfill').head()


# #### （b）填充中的对齐特性

# In[84]:


df_f = pd.DataFrame({'A':[1,3,np.nan],'B':[2,4,np.nan],'C':[3,5,np.nan]})
df_f.fillna(df_f.mean())


# #### 返回的结果中没有C，根据对齐特点不会被填充

# In[85]:


df_f.fillna(df_f.mean()[['A','B']])


# ### 2. dropna方法

# #### （a）axis参数

# In[86]:


df_d = pd.DataFrame({'A':[np.nan,np.nan,np.nan],'B':[np.nan,3,2],'C':[3,2,1]})
df_d


# In[87]:


df_d.dropna(axis=0)


# In[88]:


df_d.dropna(axis=1)


# #### （b）how参数（可以选all或者any，表示全为缺失去除和存在缺失去除）

# In[89]:


df_d.dropna(axis=1,how='all')


# #### （c）subset参数（即在某一组列范围中搜索缺失值）

# In[90]:


df_d.dropna(axis=0,subset=['B','C'])


# ## 四、插值（interpolation）

# ### 1. 线性插值

# #### （a）索引无关的线性插值
# #### 默认状态下，interpolate会对缺失的值进行线性插值

# In[91]:


s = pd.Series([1,10,15,-5,-2,np.nan,np.nan,28])
s


# In[92]:


s.interpolate()


# In[93]:


s.interpolate().plot()


# #### 此时的插值与索引无关

# In[94]:


s.index = np.sort(np.random.randint(50,300,8))
s.interpolate()
#值不变


# In[95]:


s.interpolate().plot()
#后面三个点不是线性的（如果几乎为线性函数，请重新运行上面的一个代码块，这是随机性导致的）


# #### （b）与索引有关的插值
# #### method中的index和time选项可以使插值线性地依赖索引，即插值为索引的线性函数

# In[96]:


s.interpolate(method='index').plot()
#可以看到与上面的区别


# #### 如果索引是时间，那么可以按照时间长短插值，对于时间序列将在第9章详细介绍

# In[97]:


s_t = pd.Series([0,np.nan,10]
        ,index=[pd.Timestamp('2012-05-01'),pd.Timestamp('2012-05-07'),pd.Timestamp('2012-06-03')])
s_t


# In[98]:


s_t.interpolate().plot()


# In[99]:


s_t.interpolate(method='time').plot()


# ### 2. 高级插值方法
# #### 此处的高级指的是与线性插值相比较，例如样条插值、多项式插值、阿基玛插值等（需要安装Scipy），方法详情请看[这里](https://pandas.pydata.org/pandas-docs/version/1.0.0/reference/api/pandas.DataFrame.interpolate.html#pandas.DataFrame.interpolate)
# #### 关于这部分仅给出一个官方的例子，因为插值方法是数值分析的内容，而不是Pandas中的基本知识：

# In[100]:


ser = pd.Series(np.arange(1, 10.1, .25) ** 2 + np.random.randn(37))
missing = np.array([4, 13, 14, 15, 16, 17, 18, 20, 29])
ser[missing] = np.nan
methods = ['linear', 'quadratic', 'cubic']
df = pd.DataFrame({m: ser.interpolate(method=m) for m in methods})
df.plot()


# ### 3. interpolate中的限制参数
# #### （a）limit表示最多插入多少个

# In[101]:


s = pd.Series([1,np.nan,np.nan,np.nan,5])
s.interpolate(limit=2)


# #### （b）limit_direction表示插值方向，可选forward,backward,both，默认前向

# In[102]:


s = pd.Series([np.nan,np.nan,1,np.nan,np.nan,np.nan,5,np.nan,np.nan,])
s.interpolate(limit_direction='backward')


# #### （c）limit_area表示插值区域，可选inside,outside，默认None

# In[103]:


s = pd.Series([np.nan,np.nan,1,np.nan,np.nan,np.nan,5,np.nan,np.nan,])
s.interpolate(limit_area='inside')


# In[104]:


s = pd.Series([np.nan,np.nan,1,np.nan,np.nan,np.nan,5,np.nan,np.nan,])
s.interpolate(limit_area='outside')


# ## 五、问题与练习

# ### 1. 问题

# #### 【问题一】 如何删除缺失值占比超过25%的列？
# #### 【问题二】 什么是Nullable类型？请谈谈为什么要引入这个设计？
# #### 【问题三】 对于一份有缺失值的数据，可以采取哪些策略或方法深化对它的了解？

# ### 2. 练习

# #### 【练习一】现有一份虚拟数据集，列类型分别为string/浮点/整型，请解决如下问题：
# #### （a）请以列类型读入数据，并选出C为缺失值的行。
# #### （b）现需要将A中的部分单元转为缺失值，单元格中的最小转换概率为25%，且概率大小与所在行B列单元的值成正比。

# In[105]:


pd.read_csv('data/Missing_data_one.csv').head()


# #### 【练习二】 现有一份缺失的数据集，记录了36个人来自的地区、身高、体重、年龄和工资，请解决如下问题：
# #### （a）统计各列缺失的比例并选出在后三列中至少有两个非缺失值的行。
# #### （b）请结合身高列和地区列中的数据，对体重进行合理插值。

# In[106]:


pd.read_csv('data/Missing_data_two.csv').head()

