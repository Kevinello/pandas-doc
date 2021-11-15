#!/usr/bin/env python
# coding: utf-8

# # 第8章 分类数据

# In[1]:


import pandas as pd
import numpy as np
df = pd.read_csv('data/table.csv')
df.head()


# ## 一、category的创建及其性质
# ### 1. 分类变量的创建
# #### （a）用Series创建

# In[2]:


pd.Series(["a", "b", "c", "a"], dtype="category")


# #### （b）对DataFrame指定类型创建

# In[3]:


temp_df = pd.DataFrame({'A':pd.Series(["a", "b", "c", "a"], dtype="category"),'B':list('abcd')})
temp_df.dtypes


# #### （c）利用内置Categorical类型创建

# In[4]:


cat = pd.Categorical(["a", "b", "c", "a"], categories=['a','b','c'])
pd.Series(cat)


# #### （d）利用cut函数创建

# #### 默认使用区间类型为标签

# In[5]:


pd.cut(np.random.randint(0,60,5), [0,10,30,60])


# #### 可指定字符为标签

# In[6]:


pd.cut(np.random.randint(0,60,5), [0,10,30,60], right=False, labels=['0-10','10-30','30-60'])


# ### 2. 分类变量的结构
# #### 一个分类变量包括三个部分，元素值（values）、分类类别（categories）、是否有序（order）
# #### 从上面可以看出，使用cut函数创建的分类变量默认为有序分类变量
# #### 下面介绍如何获取或修改这些属性
# #### （a）describe方法
# #### 该方法描述了一个分类序列的情况，包括非缺失值个数、元素值类别数（不是分类类别数）、最多次出现的元素及其频数

# In[7]:


s = pd.Series(pd.Categorical(["a", "b", "c", "a",np.nan], categories=['a','b','c','d']))
s.describe()


# #### （b）categories和ordered属性
# #### 查看分类类别和是否排序

# In[8]:


s.cat.categories


# In[9]:


s.cat.ordered


# ### 3. 类别的修改

# #### （a）利用set_categories修改
# #### 修改分类，但本身值不会变化

# In[10]:


s = pd.Series(pd.Categorical(["a", "b", "c", "a",np.nan], categories=['a','b','c','d']))
s.cat.set_categories(['new_a','c'])


# #### （b）利用rename_categories修改
# #### 需要注意的是该方法会把值和分类同时修改

# In[11]:


s = pd.Series(pd.Categorical(["a", "b", "c", "a",np.nan], categories=['a','b','c','d']))
s.cat.rename_categories(['new_%s'%i for i in s.cat.categories])


# #### 利用字典修改值

# In[12]:


s.cat.rename_categories({'a':'new_a','b':'new_b'})


# #### （c）利用add_categories添加

# In[13]:


s = pd.Series(pd.Categorical(["a", "b", "c", "a",np.nan], categories=['a','b','c','d']))
s.cat.add_categories(['e'])


# #### （d）利用remove_categories移除

# In[14]:


s = pd.Series(pd.Categorical(["a", "b", "c", "a",np.nan], categories=['a','b','c','d']))
s.cat.remove_categories(['d'])


# #### （e）删除元素值未出现的分类类型

# In[15]:


s = pd.Series(pd.Categorical(["a", "b", "c", "a",np.nan], categories=['a','b','c','d']))
s.cat.remove_unused_categories()


# ## 二、分类变量的排序
# #### 前面提到，分类数据类型被分为有序和无序，这非常好理解，例如分数区间的高低是有序变量，考试科目的类别一般看做无序变量

# ### 1. 序的建立

# #### （a）一般来说会将一个序列转为有序变量，可以利用as_ordered方法

# In[16]:


s = pd.Series(["a", "d", "c", "a"]).astype('category').cat.as_ordered()
s


# #### 退化为无序变量，只需要使用as_unordered

# In[17]:


s.cat.as_unordered()


# #### （b）利用set_categories方法中的order参数

# In[18]:


pd.Series(["a", "d", "c", "a"]).astype('category').cat.set_categories(['a','c','d'],ordered=True)


# #### （c）利用reorder_categories方法
# #### 这个方法的特点在于，新设置的分类必须与原分类为同一集合

# In[19]:


s = pd.Series(["a", "d", "c", "a"]).astype('category')
s.cat.reorder_categories(['a','c','d'],ordered=True)


# In[20]:


#s.cat.reorder_categories(['a','c'],ordered=True) #报错
#s.cat.reorder_categories(['a','c','d','e'],ordered=True) #报错


# ### 2. 排序

# #### 先前在第1章介绍的值排序和索引排序都是适用的

# In[21]:


s = pd.Series(np.random.choice(['perfect','good','fair','bad','awful'],50)).astype('category')
s.cat.set_categories(['perfect','good','fair','bad','awful'][::-1],ordered=True).head()


# In[22]:


s.sort_values(ascending=False).head()


# In[23]:


df_sort = pd.DataFrame({'cat':s.values,'value':np.random.randn(50)}).set_index('cat')
df_sort.head()


# In[24]:


df_sort.sort_index().head()


# ## 三、分类变量的比较操作

# ### 1. 与标量或等长序列的比较

# #### （a）标量比较

# In[25]:


s = pd.Series(["a", "d", "c", "a"]).astype('category')
s == 'a'


# #### （b）等长序列比较

# In[26]:


s == list('abcd')


# ### 2. 与另一分类变量的比较

# #### （a）等式判别（包含等号和不等号）
# #### 两个分类变量的等式判别需要满足分类完全相同

# In[27]:


s = pd.Series(["a", "d", "c", "a"]).astype('category')
s == s


# In[28]:


s != s


# In[29]:


s_new = s.cat.set_categories(['a','d','e'])
#s == s_new #报错


# #### （b）不等式判别（包含>=,<=,<,>）
# #### 两个分类变量的不等式判别需要满足两个条件：① 分类完全相同 ② 排序完全相同

# In[30]:


s = pd.Series(["a", "d", "c", "a"]).astype('category')
#s >= s #报错


# In[31]:


s = pd.Series(["a", "d", "c", "a"]).astype('category').cat.reorder_categories(['a','c','d'],ordered=True)
s >= s


# ## 四、问题与练习

# #### 【问题一】 如何使用union_categoricals方法？它的作用是什么？
# #### 【问题二】 利用concat方法将两个序列纵向拼接，它的结果一定是分类变量吗？什么情况下不是？
# #### 【问题三】 当使用groupby方法或者value_counts方法时，分类变量的统计结果和普通变量有什么区别？
# #### 【问题四】 下面的代码说明了Series创建分类变量的什么“缺陷”？如何避免？（提示：使用Series中的copy参数）

# In[32]:


cat = pd.Categorical([1, 2, 3, 10], categories=[1, 2, 3, 4, 10])
s = pd.Series(cat, name="cat")
cat


# In[33]:


s.iloc[0:2] = 10
cat


# #### 【练习一】 现继续使用第四章中的地震数据集，请解决以下问题：
# #### （a）现在将深度分为七个等级：[0,5,10,15,20,30,50,np.inf]，请以深度等级Ⅰ,Ⅱ,Ⅲ,Ⅳ,Ⅴ,Ⅵ,Ⅶ为索引并按照由浅到深的顺序进行排序。
# #### （b）在（a）的基础上，将烈度分为4个等级：[0,3,4,5,np.inf]，依次对南部地区的深度和烈度等级建立多级索引排序。

# In[34]:


pd.read_csv('data/Earthquake.csv').head()


# #### 【练习二】 对于分类变量而言，调用第4章中的变形函数会出现一个BUG（目前的版本下还未修复）：例如对于crosstab函数，按照[官方文档的说法](https://pandas.pydata.org/pandas-docs/version/1.0.0/user_guide/reshaping.html#cross-tabulations)，即使没有出现的变量也会在变形后的汇总结果中出现，但事实上并不是这样，比如下面的例子就缺少了原本应该出现的行'c'和列'f'。基于这一问题，请尝试设计my_crosstab函数，在功能上能够返回正确的结果。

# In[35]:


foo = pd.Categorical(['a', 'b'], categories=['a', 'b', 'c'])
bar = pd.Categorical(['d', 'e'], categories=['d', 'e', 'f'])
pd.crosstab(foo, bar)

