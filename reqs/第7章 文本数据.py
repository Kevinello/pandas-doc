#!/usr/bin/env python
# coding: utf-8

# # 第7章 文本数据

# In[1]:


import pandas as pd
import numpy as np


# ## 一、string类型的性质

# ### 1. string与object的区别
# #### string类型和object不同之处有三：
# #### ① 字符存取方法（string accessor methods，如str.count）会返回相应数据的Nullable类型，而object会随缺失值的存在而改变返回类型
# #### ② 某些Series方法不能在string上使用，例如： Series.str.decode()，因为存储的是字符串而不是字节
# #### ③ string类型在缺失值存储或运算时，类型会广播为pd.NA，而不是浮点型np.nan
# #### 其余全部内容在当前版本下完全一致，但迎合Pandas的发展模式，我们仍然全部用string来操作字符串

# ### 2. string类型的转换
# #### 如果将一个其他类型的容器直接转换string类型可能会出错：

# In[2]:


#pd.Series([1,'1.']).astype('string') #报错
#pd.Series([1,2]).astype('string') #报错
#pd.Series([True,False]).astype('string') #报错


# #### 当下正确的方法是分两部转换，先转为str型object，在转为string类型：

# In[3]:


pd.Series([1,'1.']).astype('str').astype('string')


# In[4]:


pd.Series([1,2]).astype('str').astype('string')


# In[5]:


pd.Series([True,False]).astype('str').astype('string')


# ## 二、拆分与拼接

# ### 1. str.split方法
# #### （a）分割符与str的位置元素选取

# In[6]:


s = pd.Series(['a_b_c', 'c_d_e', np.nan, 'f_g_h'], dtype="string")
s


# #### 根据某一个元素分割，默认为空格

# In[7]:


s.str.split('_')


# #### 这里需要注意split后的类型是object，因为现在Series中的元素已经不是string，而包含了list，且string类型只能含有字符串

# #### 对于str方法可以进行元素的选择，如果该单元格元素是列表，那么str[i]表示取出第i个元素，如果是单个元素，则先把元素转为列表在取出

# In[8]:


s.str.split('_').str[1]


# In[9]:


pd.Series(['a_b_c', ['a','b','c']], dtype="object").str[1]
#第一个元素先转为['a','_','b','_','c']


# #### （b）其他参数
# #### expand参数控制了是否将列拆开，n参数代表最多分割多少次

# In[10]:


s.str.split('_',expand=True)


# In[11]:


s.str.split('_',n=1)


# In[12]:


s.str.split('_',expand=True,n=1)


# ### 2. str.cat方法
# #### （a）不同对象的拼接模式
# #### cat方法对于不同对象的作用结果并不相同，其中的对象包括：单列、双列、多列
# #### ① 对于单个Series而言，就是指所有的元素进行字符合并为一个字符串

# In[13]:


s = pd.Series(['ab',None,'d'],dtype='string')
s


# In[14]:


s.str.cat()


# #### 其中可选sep分隔符参数，和缺失值替代字符na_rep参数

# In[15]:


s.str.cat(sep=',')


# In[16]:


s.str.cat(sep=',',na_rep='*')


# #### ② 对于两个Series合并而言，是对应索引的元素进行合并

# In[17]:


s2 = pd.Series(['24',None,None],dtype='string')
s2


# In[18]:


s.str.cat(s2)


# #### 同样也有相应参数，需要注意的是两个缺失值会被同时替换

# In[19]:


s.str.cat(s2,sep=',',na_rep='*')


# #### ③ 多列拼接可以分为表的拼接和多Series拼接

# #### 表的拼接

# In[20]:


s.str.cat(pd.DataFrame({0:['1','3','5'],1:['5','b',None]},dtype='string'),na_rep='*')


# #### 多个Series拼接

# In[21]:


s.str.cat([s+'0',s*2])


# #### （b）cat中的索引对齐
# #### 当前版本中，如果两边合并的索引不相同且未指定join参数，默认为左连接，设置join='left'

# In[22]:


s2 = pd.Series(list('abc'),index=[1,2,3],dtype='string')
s2


# In[23]:


s.str.cat(s2,na_rep='*')


# ## 三、替换
# #### 广义上的替换，就是指str.replace函数的应用，fillna是针对缺失值的替换，上一章已经提及
# #### 提到替换，就不可避免地接触到正则表达式，这里默认读者已掌握常见正则表达式知识点，若对其还不了解的，可以通过[这份资料](https://regexone.com/)来熟悉

# ### 1. str.replace的常见用法

# In[24]:


s = pd.Series(['A', 'B', 'C', 'Aaba', 'Baca','', np.nan, 'CABA', 'dog', 'cat'],dtype="string")
s


# #### 第一个值写r开头的正则表达式，后一个写替换的字符串

# In[25]:


s.str.replace(r'^[AB]','***')


# ### 2. 子组与函数替换

# #### 通过正整数调用子组（0返回字符本身，从1开始才是子组）

# In[26]:


s.str.replace(r'([ABC])(\w+)',lambda x:x.group(2)[1:]+'*')


# #### 利用?P<....>表达式可以对子组命名调用

# In[27]:


s.str.replace(r'(?P<one>[ABC])(?P<two>\w+)',lambda x:x.group('two')[1:]+'*')


# ### 3. 关于str.replace的注意事项
# #### 首先，要明确str.replace和replace并不是一个东西：
# #### str.replace针对的是object类型或string类型，默认是以正则表达式为操作，目前暂时不支持DataFrame上使用
# #### replace针对的是任意类型的序列或数据框，如果要以正则表达式替换，需要设置regex=True，该方法通过字典可支持多列替换
# #### 但现在由于string类型的初步引入，用法上出现了一些问题，这些issue有望在以后的版本中修复
# #### （a）str.replace赋值参数不得为pd.NA
# #### 这听上去非常不合理，例如对满足某些正则条件的字符串替换为缺失值，直接更改为缺失值在当下版本就会报错

# In[28]:


#pd.Series(['A','B'],dtype='string').str.replace(r'[A]',pd.NA) #报错
#pd.Series(['A','B'],dtype='O').str.replace(r'[A]',pd.NA) #报错


# #### 此时，可以先转为object类型再转换回来，曲线救国：

# In[29]:


pd.Series(['A','B'],dtype='string').astype('O').replace(r'[A]',pd.NA,regex=True).astype('string')


# #### 至于为什么不用replace函数的regex替换（但string类型replace的非正则替换是可以的），原因在下面一条

# #### （b）对于string类型Series，在使用replace函数时不能使用正则表达式替换
# #### 该bug现在还未修复

# In[30]:


pd.Series(['A','B'],dtype='string').replace(r'[A]','C',regex=True)


# In[31]:


pd.Series(['A','B'],dtype='O').replace(r'[A]','C',regex=True)


# #### （c）string类型序列如果存在缺失值，不能使用replace替换

# In[32]:


#pd.Series(['A',np.nan],dtype='string').replace('A','B') #报错


# In[33]:


pd.Series(['A',np.nan],dtype='string').str.replace('A','B')


# #### 综上，概况的说，除非需要赋值元素为缺失值（转为object再转回来），否则请使用str.replace方法

# ## 四、子串匹配与提取

# ### 1. str.extract方法
# #### （a）常见用法

# In[34]:


pd.Series(['10-87', '10-88', '10-89'],dtype="string").str.extract(r'([\d]{2})-([\d]{2})')


# #### 使用子组名作为列名

# In[35]:


pd.Series(['10-87', '10-88', '-89'],dtype="string").str.extract(r'(?P<name_1>[\d]{2})-(?P<name_2>[\d]{2})')


# #### 利用?正则标记选择部分提取

# In[36]:


pd.Series(['10-87', '10-88', '-89'],dtype="string").str.extract(r'(?P<name_1>[\d]{2})?-(?P<name_2>[\d]{2})')


# In[37]:


pd.Series(['10-87', '10-88', '10-'],dtype="string").str.extract(r'(?P<name_1>[\d]{2})-(?P<name_2>[\d]{2})?')


# #### （b）expand参数（默认为True）

# #### 对于一个子组的Series，如果expand设置为False，则返回Series，若大于一个子组，则expand参数无效，全部返回DataFrame
# #### 对于一个子组的Index，如果expand设置为False，则返回提取后的Index，若大于一个子组且expand为False，报错

# In[38]:


s = pd.Series(["a1", "b2", "c3"], ["A11", "B22", "C33"], dtype="string")
s.index


# In[39]:


s.str.extract(r'([\w])')


# In[40]:


s.str.extract(r'([\w])',expand=False)


# In[41]:


s.index.str.extract(r'([\w])')


# In[42]:


s.index.str.extract(r'([\w])',expand=False)


# In[43]:


s.index.str.extract(r'([\w])([\d])')


# In[44]:


#s.index.str.extract(r'([\w])([\d])',expand=False) #报错


# ### 2. str.extractall方法

# #### 与extract只匹配第一个符合条件的表达式不同，extractall会找出所有符合条件的字符串，并建立多级索引（即使只找到一个）

# In[45]:


s = pd.Series(["a1a2", "b1", "c1"], index=["A", "B", "C"],dtype="string")
two_groups = '(?P<letter>[a-z])(?P<digit>[0-9])'
s.str.extract(two_groups, expand=True)


# In[46]:


s.str.extractall(two_groups)


# In[47]:


s['A']='a1'
s.str.extractall(two_groups)


# #### 如果想查看第i层匹配，可使用xs方法

# In[48]:


s = pd.Series(["a1a2", "b1b2", "c1c2"], index=["A", "B", "C"],dtype="string")
s.str.extractall(two_groups).xs(1,level='match')


# ### 3. str.contains和str.match
# #### 前者的作用为检测是否包含某种正则模式

# In[49]:


pd.Series(['1', None, '3a', '3b', '03c'], dtype="string").str.contains(r'[0-9][a-z]')


# #### 可选参数为na

# In[50]:


pd.Series(['1', None, '3a', '3b', '03c'], dtype="string").str.contains('a', na=False)


# #### str.match与其区别在于，match依赖于python的re.match，检测内容为是否从头开始包含该正则模式

# In[51]:


pd.Series(['1', None, '3a_', '3b', '03c'], dtype="string").str.match(r'[0-9][a-z]',na=False)


# In[52]:


pd.Series(['1', None, '_3a', '3b', '03c'], dtype="string").str.match(r'[0-9][a-z]',na=False)


# ## 五、常用字符串方法

# ### 1. 过滤型方法
# #### （a）str.strip
# #### 常用于过滤空格

# In[53]:


pd.Series(list('abc'),index=[' space1  ','space2  ','  space3'],dtype="string").index.str.strip()


# #### （b）str.lower和str.upper

# In[54]:


pd.Series('A',dtype="string").str.lower()


# In[55]:


pd.Series('a',dtype="string").str.upper()


# #### （c）str.swapcase和str.capitalize
# #### 分别表示交换字母大小写和大写首字母

# In[56]:


pd.Series('abCD',dtype="string").str.swapcase()


# In[57]:


pd.Series('abCD',dtype="string").str.capitalize()


# ### 2. isnumeric方法
# #### 检查每一位是否都是数字，请问如何判断是否是数值？（问题二）

# In[58]:


pd.Series(['1.2','1','-0.3','a',np.nan],dtype="string").str.isnumeric()


# ## 六、问题与练习
# ### 1. 问题

# #### 【问题一】 str对象方法和df/Series对象方法有什么区别？
# #### 【问题二】 给出一列string类型，如何判断单元格是否是数值型数据？
# #### 【问题三】 rsplit方法的作用是什么？它在什么场合下适用？
# #### 【问题四】 在本章的第二到第四节分别介绍了字符串类型的5类操作，请思考它们各自应用于什么场景？

# ### 2. 练习
# #### 【练习一】 现有一份关于字符串的数据集，请解决以下问题：
# #### （a）现对字符串编码存储人员信息（在编号后添加ID列），使用如下格式：“×××（名字）：×国人，性别×，生于×年×月×日”
# #### （b）将（a）中的人员生日信息部分修改为用中文表示（如一九七四年十月二十三日），其余返回格式不变。
# #### （c）将（b）中的ID列结果拆分为原列表相应的5列，并使用equals检验是否一致。

# In[59]:


pd.read_csv('data/String_data_one.csv',index_col='人员编号').head()


# #### 【练习二】 现有一份半虚拟的数据集，第一列包含了新型冠状病毒的一些新闻标题，请解决以下问题：
# #### （a）选出所有关于北京市和上海市新闻标题的所在行。
# #### （b）求col2的均值。
# #### （c）求col3的均值。

# In[60]:


pd.read_csv('data/String_data_two.csv').head()

