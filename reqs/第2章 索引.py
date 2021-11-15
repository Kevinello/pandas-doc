#!/usr/bin/env python
# coding: utf-8

# # 第2章 索引

# In[1]:


import numpy as np
import pandas as pd
df = pd.read_csv('data/table.csv',index_col='ID')
df.head()


# ## 一、单级索引
# ### 1. loc方法、iloc方法、[]操作符
# #### 最常用的索引方法可能就是这三类，其中iloc表示位置索引，loc表示标签索引，[]也具有很大的便利性，各有特点
# #### （a）loc方法
# #### ① 单行索引：

# In[2]:


df.loc[1103]


# #### ② 多行索引：

# In[3]:


df.loc[[1102,2304]]


# #### （注意：所有在loc中使用的切片全部包含右端点！这是因为如果作为Pandas的使用者，那么肯定不太关心最后一个标签再往后一位是什么，但是如果是左闭右开，那么就很麻烦，先要知道再后面一列的名字是什么，非常不方便，因此Pandas中将loc设计为左右全闭）

# In[4]:


df.loc[1304:2103].head()


# In[5]:


df.loc[2402::-1].head()


# #### ③ 单列索引：

# In[6]:


df.loc[:,'Height'].head()


# #### ④ 多列索引：

# In[7]:


df.loc[:,['Height','Math']].head()


# In[8]:


df.loc[:,'Height':'Math'].head()


# #### ⑤ 联合索引：

# In[9]:


df.loc[1102:2401:3,'Height':'Math'].head()


# #### ⑥ 函数式索引：

# In[10]:


df.loc[lambda x:x['Gender']=='M'].head()
#loc中使用的函数，传入参数就是前面的df


# In[11]:


#这里的例子表示，loc中能够传入函数，并且函数的输入值是整张表，输出为标量、切片、合法列表（元素出现在索引中）、合法索引
def f(x):
    return [1101,1103]
df.loc[f]


# #### ⑦ 布尔索引（将重点在第2节介绍）

# In[12]:


df.loc[df['Address'].isin(['street_7','street_4'])].head()


# In[13]:


df.loc[[True if i[-1]=='4' or i[-1]=='7' else False for i in df['Address'].values]].head()


# #### 小节：本质上说，loc中能传入的只有布尔列表和索引子集构成的列表，只要把握这个原则就很容易理解上面那些操作
# #### （b）iloc方法（注意与loc不同，切片右端点不包含）
# #### ① 单行索引：

# In[14]:


df.iloc[3]


# #### ② 多行索引：

# In[15]:


df.iloc[3:5]


# #### ③ 单列索引：

# In[16]:


df.iloc[:,3].head()


# #### ④ 多列索引：

# In[17]:


df.iloc[:,7::-2].head()


# #### ⑤ 混合索引：

# In[18]:


df.iloc[3::4,7::-2].head()


# #### ⑥ 函数式索引：

# In[19]:


df.iloc[lambda x:[3]].head()


# #### 小节：iloc中接收的参数只能为整数或整数列表或布尔列表，不能使用布尔Series，如果要用就必须如下把values拿出来

# In[20]:


#df.iloc[df['School']=='S_1'].head() #报错
df.iloc[(df['School']=='S_1').values].head()


# #### （c） []操作符
# #### （c.1）Series的[]操作
# #### ① 单元素索引：

# In[21]:


s = pd.Series(df['Math'],index=df.index)
s[1101]
#使用的是索引标签


# #### ② 多行索引：

# In[22]:


s[0:4]
#使用的是绝对位置的整数切片，与元素无关，这里容易混淆


# #### ③ 函数式索引：

# In[23]:


s[lambda x: x.index[16::-6]]
#注意使用lambda函数时，直接切片(如：s[lambda x: 16::-6])就报错，此时使用的不是绝对位置切片，而是元素切片，非常易错


# #### ④ 布尔索引：

# In[24]:


s[s>80]


# #### 【注意】如果不想陷入困境，请不要在行索引为浮点时使用[]操作符，因为在Series中[]的浮点切片并不是进行位置比较，而是值比较，非常特殊

# In[25]:


s_int = pd.Series([1,2,3,4],index=[1,3,5,6])
s_float = pd.Series([1,2,3,4],index=[1.,3.,5.,6.])
s_int


# In[26]:


s_int[2:]


# In[27]:


s_float


# In[28]:


#注意和s_int[2:]结果不一样了，因为2这里是元素而不是位置
s_float[2:]


# #### （c.2）DataFrame的[]操作
# #### ① 单行索引：

# In[29]:


df[1:2]
#这里非常容易写成df['label']，会报错
#同Series使用了绝对位置切片
#如果想要获得某一个元素，可用如下get_loc方法：


# In[30]:


row = df.index.get_loc(1102)
df[row:row+1]


# #### ② 多行索引：

# In[31]:


#用切片，如果是选取指定的某几行，推荐使用loc，否则很可能报错
df[3:5]


# #### ③ 单列索引：

# In[32]:


df['School'].head()


# #### ④ 多列索引：

# In[33]:


df[['School','Math']].head()


# #### ⑤函数式索引：

# In[34]:


df[lambda x:['Math','Physics']].head()


# #### ⑥ 布尔索引：

# In[35]:


df[df['Gender']=='F'].head()


# #### 小节：一般来说，[]操作符常用于列选择或布尔选择，尽量避免行的选择
# ### 2. 布尔索引
# #### （a）布尔符号：'&','|','~'：分别代表和and，或or，取反not

# In[36]:


df[(df['Gender']=='F')&(df['Address']=='street_2')].head()


# In[37]:


df[(df['Math']>85)|(df['Address']=='street_7')].head()


# In[38]:


df[~((df['Math']>75)|(df['Address']=='street_1'))].head()


# #### loc和[]中相应位置都能使用布尔列表选择：

# In[39]:


df.loc[df['Math']>60,df.columns=='Physics'].head()
#思考：为什么df.loc[df['Math']>60,(df[:8]['Address']=='street_6').values].head()得到和上述结果一样？values能去掉吗？


# #### （b） isin方法

# In[40]:


df[df['Address'].isin(['street_1','street_4'])&df['Physics'].isin(['A','A+'])]


# In[41]:


#上面也可以用字典方式写：
df[df[['Address','Physics']].isin({'Address':['street_1','street_4'],'Physics':['A','A+']}).all(1)]
#all与&的思路是类似的，其中的1代表按照跨列方向判断是否全为True


# ### 3. 快速标量索引
# #### 当只需要取一个元素时，at和iat方法能够提供更快的实现：

# In[42]:


display(df.at[1101,'School'])
display(df.loc[1101,'School'])
display(df.iat[0,0])
display(df.iloc[0,0])
#可尝试去掉注释对比时间
#%timeit df.at[1101,'School']
#%timeit df.loc[1101,'School']
#%timeit df.iat[0,0]
#%timeit df.iloc[0,0]


# ### 4. 区间索引
# #### 此处介绍并不是说只能在单级索引中使用区间索引，只是作为一种特殊类型的索引方式，在此处先行介绍
# #### （a）利用interval_range方法

# In[43]:


pd.interval_range(start=0,end=5)
#closed参数可选'left''right''both''neither'，默认左开右闭


# In[44]:


pd.interval_range(start=0,periods=8,freq=5)
#periods参数控制区间个数，freq控制步长


# #### （b）利用cut将数值列转为区间为元素的分类变量，例如统计数学成绩的区间情况：

# In[45]:


math_interval = pd.cut(df['Math'],bins=[0,40,60,80,100])
#注意，如果没有类型转换，此时并不是区间类型，而是category类型
math_interval.head()


# #### （c）区间索引的选取

# In[46]:


df_i = df.join(math_interval,rsuffix='_interval')[['Math','Math_interval']]            .reset_index().set_index('Math_interval')
df_i.head()


# In[47]:


df_i.loc[65].head()
#包含该值就会被选中


# In[48]:


df_i.loc[[65,90]].head()


# #### 如果想要选取某个区间，先要把分类变量转为区间变量，再使用overlap方法：

# In[49]:


#df_i.loc[pd.Interval(70,75)].head() 报错
df_i[df_i.index.astype('interval').overlaps(pd.Interval(70, 85))].head()


# ## 二、多级索引
# ### 1. 创建多级索引
# #### （a）通过from_tuple或from_arrays
# #### ① 直接创建元组

# In[50]:


tuples = [('A','a'),('A','b'),('B','a'),('B','b')]
mul_index = pd.MultiIndex.from_tuples(tuples, names=('Upper', 'Lower'))
mul_index


# In[51]:


pd.DataFrame({'Score':['perfect','good','fair','bad']},index=mul_index)


# #### ② 利用zip创建元组

# In[52]:


L1 = list('AABB')
L2 = list('abab')
tuples = list(zip(L1,L2))
mul_index = pd.MultiIndex.from_tuples(tuples, names=('Upper', 'Lower'))
pd.DataFrame({'Score':['perfect','good','fair','bad']},index=mul_index)


# #### ③ 通过Array创建

# In[53]:


arrays = [['A','a'],['A','b'],['B','a'],['B','b']]
mul_index = pd.MultiIndex.from_tuples(arrays, names=('Upper', 'Lower'))
pd.DataFrame({'Score':['perfect','good','fair','bad']},index=mul_index)


# In[54]:


mul_index
#由此看出内部自动转成元组


# #### （b）通过from_product

# In[55]:


L1 = ['A','B']
L2 = ['a','b']
pd.MultiIndex.from_product([L1,L2],names=('Upper', 'Lower'))
#两两相乘


# #### （c）指定df中的列创建（set_index方法）

# In[56]:


df_using_mul = df.set_index(['Class','Address'])
df_using_mul.head()


# ### 2. 多层索引切片

# In[57]:


df_using_mul.head()


# #### （a）一般切片

# In[58]:


#df_using_mul.loc['C_2','street_5']
#当索引不排序时，单个索引会报出性能警告
#df_using_mul.index.is_lexsorted()
#该函数检查是否排序
df_using_mul.sort_index().loc['C_2','street_5']
#df_using_mul.sort_index().index.is_lexsorted()


# In[59]:


#df_using_mul.loc[('C_2','street_5'):] 报错
#当不排序时，不能使用多层切片
df_using_mul.sort_index().loc[('C_2','street_6'):('C_3','street_4')]
#注意此处由于使用了loc，因此仍然包含右端点


# In[60]:


df_using_mul.sort_index().loc[('C_2','street_7'):'C_3'].head()
#非元组也是合法的，表示选中该层所有元素


# #### （b）第一类特殊情况：由元组构成列表

# In[61]:


df_using_mul.sort_index().loc[[('C_2','street_7'),('C_3','street_2')]]
#表示选出某几个元素，精确到最内层索引


# #### （c）第二类特殊情况：由列表构成元组

# In[62]:


df_using_mul.sort_index().loc[(['C_2','C_3'],['street_4','street_7']),:]
#选出第一层在‘C_2’和'C_3'中且第二层在'street_4'和'street_7'中的行


# ### 3. 多层索引中的slice对象

# In[63]:


L1,L2 = ['A','B','C'],['a','b','c']
mul_index1 = pd.MultiIndex.from_product([L1,L2],names=('Upper', 'Lower'))
L3,L4 = ['D','E','F'],['d','e','f']
mul_index2 = pd.MultiIndex.from_product([L3,L4],names=('Big', 'Small'))
df_s = pd.DataFrame(np.random.rand(9,9),index=mul_index1,columns=mul_index2)
df_s


# In[64]:


idx=pd.IndexSlice


# #### 索引Slice的使用非常灵活：

# In[65]:


df_s.loc[idx['B':,df_s['D']['d']>0.3],idx[df_s.sum()>4]]
#df_s.sum()默认为对列求和，因此返回一个长度为9的数值列表


# ### 4. 索引层的交换
# #### （a）swaplevel方法（两层交换）

# In[66]:


df_using_mul.head()


# In[67]:


df_using_mul.swaplevel(i=1,j=0,axis=0).sort_index().head()


# #### （b）reorder_levels方法（多层交换）

# In[68]:


df_muls = df.set_index(['School','Class','Address'])
df_muls.head()


# In[69]:


df_muls.reorder_levels([2,0,1],axis=0).sort_index().head()


# In[70]:


#如果索引有name，可以直接使用name
df_muls.reorder_levels(['Address','School','Class'],axis=0).sort_index().head()


# ## 三、索引设定
# ### 1. index_col参数
# #### index_col是read_csv中的一个参数，而不是某一个方法：

# In[71]:


pd.read_csv('data/table.csv',index_col=['Address','School']).head()


# ### 2. reindex和reindex_like
# #### reindex是指重新索引，它的重要特性在于索引对齐，很多时候用于重新排序

# In[72]:


df.head()


# In[73]:


df.reindex(index=[1101,1203,1206,2402])


# In[74]:


df.reindex(columns=['Height','Gender','Average']).head()


# #### 可以选择缺失值的填充方法：fill_value和method（bfill/ffill/nearest），其中method参数必须索引单调

# In[75]:


df.reindex(index=[1101,1203,1206,2402],method='bfill')
#bfill表示用所在索引1206的后一个有效行填充，ffill为前一个有效行，nearest是指最近的


# In[76]:


df.reindex(index=[1101,1203,1206,2402],method='nearest')
#数值上1205比1301更接近1206，因此用前者填充


# #### reindex_like的作用为生成一个横纵索引完全与参数列表一致的DataFrame，数据使用被调用的表

# In[77]:


df_temp = pd.DataFrame({'Weight':np.zeros(5),
                        'Height':np.zeros(5),
                        'ID':[1101,1104,1103,1106,1102]}).set_index('ID')
df_temp.reindex_like(df[0:5][['Weight','Height']])


# #### 如果df_temp单调还可以使用method参数：

# In[78]:


df_temp = pd.DataFrame({'Weight':range(5),
                        'Height':range(5),
                        'ID':[1101,1104,1103,1106,1102]}).set_index('ID').sort_index()
df_temp.reindex_like(df[0:5][['Weight','Height']],method='bfill')
#可以自行检验这里的1105的值是否是由bfill规则填充


# ### 3. set_index和reset_index
# #### 先介绍set_index：从字面意思看，就是将某些列作为索引

# #### 使用表内列作为索引：

# In[79]:


df.head()


# In[80]:


df.set_index('Class').head()


# #### 利用append参数可以将当前索引维持不变

# In[81]:


df.set_index('Class',append=True).head()


# #### 当使用与表长相同的列作为索引（需要先转化为Series，否则报错）：

# In[82]:


df.set_index(pd.Series(range(df.shape[0]))).head()


# #### 可以直接添加多级索引：

# In[83]:


df.set_index([pd.Series(range(df.shape[0])),pd.Series(np.ones(df.shape[0]))]).head()


# #### 下面介绍reset_index方法，它的主要功能是将索引重置
# #### 默认状态直接恢复到自然数索引：

# In[84]:


df.reset_index().head()


# #### 用level参数指定哪一层被reset，用col_level参数指定set到哪一层：

# In[85]:


L1,L2 = ['A','B','C'],['a','b','c']
mul_index1 = pd.MultiIndex.from_product([L1,L2],names=('Upper', 'Lower'))
L3,L4 = ['D','E','F'],['d','e','f']
mul_index2 = pd.MultiIndex.from_product([L3,L4],names=('Big', 'Small'))
df_temp = pd.DataFrame(np.random.rand(9,9),index=mul_index1,columns=mul_index2)
df_temp.head()


# In[86]:


df_temp1 = df_temp.reset_index(level=1,col_level=1)
df_temp1.head()


# In[87]:


df_temp1.columns
#看到的确插入了level2


# In[88]:


df_temp1.index
#最内层索引被移出


# ### 4. rename_axis和rename
# #### rename_axis是针对多级索引的方法，作用是修改某一层的索引名，而不是索引标签

# In[89]:


df_temp.rename_axis(index={'Lower':'LowerLower'},columns={'Big':'BigBig'})


# #### rename方法用于修改列或者行索引标签，而不是索引名：

# In[90]:


df_temp.rename(index={'A':'T'},columns={'e':'changed_e'}).head()


# ## 四、常用索引型函数
# ### 1. where函数
# #### 当对条件为False的单元进行填充：

# In[91]:


df.head()


# In[92]:


df.where(df['Gender']=='M').head()
#不满足条件的行全部被设置为NaN


# #### 通过这种方法筛选结果和[]操作符的结果完全一致：

# In[93]:


df.where(df['Gender']=='M').dropna().head()


# #### 第一个参数为布尔条件，第二个参数为填充值：

# In[94]:


df.where(df['Gender']=='M',np.random.rand(df.shape[0],df.shape[1])).head()


# ### 2. mask函数
# #### mask函数与where功能上相反，其余完全一致，即对条件为True的单元进行填充

# In[95]:


df.mask(df['Gender']=='M').dropna().head()


# In[96]:


df.mask(df['Gender']=='M',np.random.rand(df.shape[0],df.shape[1])).head()


# ### 3. query函数

# In[97]:


df.head()


# #### query函数中的布尔表达式中，下面的符号都是合法的：行列索引名、字符串、and/not/or/&/|/~/not in/in/==/!=、四则运算符

# In[98]:


df.query('(Address in ["street_6","street_7"])&(Weight>(70+10))&(ID in [1303,2304,2402])')


# ## 五、重复元素处理
# ### 1. duplicated方法
# #### 该方法返回了是否重复的布尔列表

# In[99]:


df.duplicated('Class').head()


# #### 可选参数keep默认为first，即首次出现设为不重复，若为last，则最后一次设为不重复，若为False，则所有重复项为True

# In[100]:


df.duplicated('Class',keep='last').tail()


# In[101]:


df.duplicated('Class',keep=False).head()


# ### 2. drop_duplicates方法
# #### 从名字上看出为剔除重复项，这在后面章节中的分组操作中可能是有用的，例如需要保留每组的第一个值：

# In[102]:


df.drop_duplicates('Class')


# #### 参数与duplicate函数类似：

# In[103]:


df.drop_duplicates('Class',keep='last')


# #### 在传入多列时等价于将多列共同视作一个多级索引，比较重复项：

# In[104]:


df.drop_duplicates(['School','Class'])


# ## 六、抽样函数
# #### 这里的抽样函数指的就是sample函数
# #### （a）n为样本量

# In[105]:


df.sample(n=5)


# #### （b）frac为抽样比

# In[106]:


df.sample(frac=0.05)


# #### （c）replace为是否放回

# In[107]:


df.sample(n=df.shape[0],replace=True).head()


# In[108]:


df.sample(n=35,replace=True).index.is_unique


# #### （d）axis为抽样维度，默认为0，即抽行

# In[109]:


df.sample(n=3,axis=1).head()


# #### （e）weights为样本权重，自动归一化

# In[110]:


df.sample(n=3,weights=np.random.rand(df.shape[0])).head()


# In[111]:


#以某一列为权重，这在抽样理论中很常见
#抽到的概率与Math数值成正比
df.sample(n=3,weights=df['Math']).head()


# ## 七、问题与练习

# ### 1. 问题
# #### 【问题一】 如何更改列或行的顺序？如何交换奇偶行（列）的顺序？
# #### 【问题二】 如果要选出DataFrame的某个子集，请给出尽可能多的方法实现。
# #### 【问题三】 query函数比其他索引方法的速度更慢吗？在什么场合使用什么索引最高效？
# #### 【问题四】 单级索引能使用Slice对象吗？能的话怎么使用，请给出一个例子。
# #### 【问题五】 如何快速找出某一列的缺失值所在索引？
# #### 【问题六】 索引设定中的所有方法分别适用于哪些场合？怎么直接把某个DataFrame的索引换成任意给定同长度的索引？
# #### 【问题七】 多级索引有什么适用场合？
# #### 【问题八】 对于多层索引，怎么对内层进行条件筛选？
# #### 【问题九】 什么时候需要重复元素处理？

# ### 2. 练习
# #### 【练习一】 现有一份关于UFO的数据集，请解决下列问题：

# In[112]:


pd.read_csv('data/UFO.csv').head()


# #### （a）在所有被观测时间超过60s的时间中，哪个形状最多？
# #### （b）对经纬度进行划分：-180°至180°以30°为一个经度划分，-90°至90°以18°为一个维度划分，请问哪个区域中报告的UFO事件数量最多？

# #### 【练习二】 现有一份关于口袋妖怪的数据集，请解决下列问题：

# In[113]:


pd.read_csv('data/Pokemon.csv').head()


# #### （a）双属性的Pokemon占总体比例的多少？
# #### （b）在所有种族值（Total）不小于580的Pokemon中，非神兽（Legendary=False）的比例为多少？
# #### （c）在第一属性为格斗系（Fighting）的Pokemon中，物攻排名前三高的是哪些？
# #### （d）请问六项种族指标（HP、物攻、特攻、物防、特防、速度）极差的均值最大的是哪个属性（只考虑第一属性，且均值是对属性而言）？
# #### （e）哪个属性（只考虑第一属性）神兽占总Pokemon的比例最高？该属性神兽的种族值也是最高的吗？
