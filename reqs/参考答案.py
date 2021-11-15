#!/usr/bin/env python
# coding: utf-8

# # 参考答案

# In[1]:


import pandas as pd
import numpy as np


# ## 第1章：练习一
# ### (a)

# In[2]:


df = pd.read_csv('data/Game_of_Thrones_Script.csv')
df.head()


# In[3]:


df['Name'].nunique()


# ### (b)

# In[4]:


df['Name'].value_counts().index[0]


# ### (c) 由于还没有学分组，因此方法繁琐

# In[5]:


df_words = df.assign(Words=df['Sentence'].apply(lambda x:len(x.split()))).sort_values(by='Name')
df_words.head()


# In[6]:


L_count = []
N_words = list(zip(df_words['Name'],df_words['Words']))
for i in N_words:
    if i == N_words[0]:
        L_count.append(i[1])
        last = i[0]
    else:
        L_count.append(L_count[-1]+i[1] if i[0]==last else i[1])
        last = i[0]
df_words['Count']=L_count
df_words['Name'][df_words['Count'].idxmax()]


# ## 第1章：练习二

# ### (a)

# In[7]:


df = pd.read_csv('data/Kobe_data.csv',index_col='shot_id')
df.head()


# In[8]:


pd.Series(list(zip(df['action_type'],df['combined_shot_type']))).value_counts().index[0]


# ### (b)

# In[9]:


pd.Series(list(list(zip(*(pd.Series(list(zip(df['game_id'],df['opponent'])))
                          .unique()).tolist()))[1])).value_counts().index[0]


# ## 第2章：练习一

# ### (a)

# In[10]:


df = pd.read_csv('data/UFO.csv')
df.rename(columns={'duration (seconds)':'duration'},inplace=True)
df['duration'].astype('float')
df.head()


# In[11]:


df.query('duration > 60')['shape'].value_counts().index[0]


# ### (b)

# In[12]:


bins_long = np.linspace(-180,180,13).tolist()
bins_la = np.linspace(-90,90,11).tolist()
cuts_long = pd.cut(df['longitude'],bins=bins_long)
df['cuts_long'] = cuts_long
cuts_la = pd.cut(df['latitude'],bins=bins_la)
df['cuts_la'] = cuts_la
df.head()


# In[13]:


df.set_index(['cuts_long','cuts_la']).index.value_counts().head()


# ## 第2章：练习二

# ### (a)

# In[14]:


df = pd.read_csv('data/Pokemon.csv')
df.head()


# In[15]:


df['Type 2'].count()/df.shape[0]


# ### (b)

# In[16]:


df.query('Total >= 580')['Legendary'].value_counts(normalize=True)


# ### (c) 分别为Maga形态的路卡利欧、修缮老头、怪力

# In[17]:


df[df['Type 1']=='Fighting'].sort_values(by='Attack',ascending=False).iloc[:3]


# ### (d) 钢系的极差均值最大

# In[18]:


df['range'] = df.iloc[:,5:11].max(axis=1)-df.iloc[:,5:11].min(axis=1)
attribute = df[['Type 1','range']].set_index('Type 1')
max_range = 0
result = ''
for i in attribute.index.unique():
    temp = attribute.loc[i,:].mean()
    if temp.values[0] > max_range:
        max_range = temp.values[0]
        result = i
result


# ### (e) 超能系占比最高，但普通系均值最高（因为创世神的关系）

# In[19]:


df.query('Legendary == True')['Type 1'].value_counts(normalize=True).index[0]


# In[20]:


attribute = df.query('Legendary == True')[['Type 1','Total']].set_index('Type 1')
max_value = 0
result = ''
for i in attribute.index.unique()[:-1]:
    temp = attribute.loc[i,:].mean()
    if temp[0] > max_value:
        max_value = temp[0]
        result = i
result


# ## 第3章：练习一

# ### (a) 极差为17561

# In[21]:


df = pd.read_csv('data/Diamonds.csv')
df.head()


# In[22]:


df_r = df.query('carat>1')['price']
df_r.max()-df_r.min()


# ### (b) 0-0.2分位数区间最多的为‘E’，其余区间都为‘G’

# In[23]:


bins = df['depth'].quantile(np.linspace(0,1,6)).tolist()
cuts = pd.cut(df['depth'],bins=bins) #可选label添加自定义标签
df['cuts'] = cuts
df.head()


# In[24]:


color_result = df.groupby('cuts')['color'].describe()
color_result


# ### 前三个分位数区间不满足条件，后两个区间中数量最多的颜色的确是均重价格中最贵的

# In[25]:


df['均重价格']=df['price']/df['carat']
color_result['top'] == [i[1] for i in df.groupby(['cuts'
                                ,'color'])['均重价格'].mean().groupby(['cuts']).idxmax().values]


# ### (c) 结果见下：

# In[26]:


df = df.drop(columns='均重价格')
cuts = pd.cut(df['carat'],bins=[0,0.5,1,1.5,2,np.inf]) #可选label添加自定义标签
df['cuts'] = cuts
df.head()


# In[27]:


def f(nums):
    if not nums:        
        return 0
    res = 1                            
    cur_len = 1                        
    for i in range(1, len(nums)):      
        if nums[i-1] < nums[i]:        
            cur_len += 1                
            res = max(cur_len, res)     
        else:                       
            cur_len = 1                 
    return res


# In[28]:


for name,group in df.groupby('cuts'):
    group = group.sort_values(by='depth')
    s = group['price']
    print(name,f(s.tolist()))


# ### (d) 计算结果如下：

# In[29]:


for name,group in df[['carat','price','color']].groupby('color'):
    L1 = np.array([np.ones(group.shape[0]),group['carat']]).reshape(2,group.shape[0])
    L2 = group['price']
    result = (np.linalg.inv(L1.dot(L1.T)).dot(L1)).dot(L2).reshape(2,1)
    print('当颜色为%s时，截距项为：%f，回归系数为：%f'%(name,result[0],result[1]))


# ## 第3章：练习二

# ### (a)

# In[30]:


df = pd.read_csv('data/Drugs.csv')
df.head()


# In[31]:


idx=pd.IndexSlice
for i in range(2010,2018):
    county = (df.groupby(['COUNTY','YYYY']).sum().loc[idx[:,i],:].idxmax()[0][0])
    state = df.query('COUNTY == "%s"'%county)['State'].iloc[0]
    state_true = df.groupby(['State','YYYY']).sum().loc[idx[:,i],:].idxmax()[0][0]
    if state==state_true:
        print('在%d年，%s县的报告数最多，它所属的州%s也是报告数最多的'%(i,county,state))
    else:
        print('在%d年，%s县的报告数最多，但它所属的州%s不是报告数最多的，%s州报告数最多'%(i,county,state,state_true))


# ### (b) OH州增加最多，Heroin是增量最大的，但增幅最大的是Acetyl fentanyl

# In[32]:


df_b = df[(df['YYYY'].isin([2014,2015]))&(df['SubstanceName']=='Heroin')]
df_add = df_b.groupby(['YYYY','State']).sum()
(df_add.loc[2015]-df_add.loc[2014]).idxmax()


# In[33]:


df_b = df[(df['YYYY'].isin([2014,2015]))&(df['State']=='OH')]
df_add = df_b.groupby(['YYYY','SubstanceName']).sum()
display((df_add.loc[2015]-df_add.loc[2014]).idxmax()) #这里利用了索引对齐的特点
display((df_add.loc[2015]/df_add.loc[2014]).idxmax())


# ## 第4章：练习一

# ### (a)

# In[34]:


df = pd.read_csv('data/Drugs.csv',index_col=['State','COUNTY']).sort_index()
df.head()


# In[35]:


result = pd.pivot_table(df,index=['State','COUNTY','SubstanceName']
                 ,columns='YYYY'
                 ,values='DrugReports',fill_value='-').reset_index().rename_axis(columns={'YYYY':''})
result.head()


# ### (b)

# In[36]:


result_melted = result.melt(id_vars=result.columns[:3],value_vars=result.columns[-8:]
                ,var_name='YYYY',value_name='DrugReports').query('DrugReports != "-"')
result2 = result_melted.sort_values(by=['State','COUNTY','YYYY'
                                    ,'SubstanceName']).reset_index().drop(columns='index')
#下面其实无关紧要，只是交换两个列再改一下类型（因为‘-’所以type变成object了）
cols = list(result2.columns)
a, b = cols.index('SubstanceName'), cols.index('YYYY')
cols[b], cols[a] = cols[a], cols[b]
result2 = result2[cols].astype({'DrugReports':'int','YYYY':'int'})
result2.head()


# In[37]:


df_tidy = df.reset_index().sort_values(by=result2.columns[:4].tolist()).reset_index().drop(columns='index')
df_tidy.head()


# In[38]:


df_tidy.equals(result2)


# ## 第4章：练习二

# ### (a)

# In[39]:


df = pd.read_csv('data/Earthquake.csv')
df = df.sort_values(by=df.columns.tolist()[:3]).sort_index(axis=1).reset_index().drop(columns='index')
df.head()


# In[40]:


result = pd.pivot_table(df,index=['日期','时间','维度','经度']
            ,columns='方向'
            ,values=['烈度','深度','距离'],fill_value='-').stack(level=0).rename_axis(index={None:'地震参数'})
result.head(6)


# ### (b)

# In[41]:


df_result = result.unstack().stack(0)[(~(result.unstack().stack(0)=='-')).any(1)].reset_index()
df_result.columns.name=None
df_result = df_result.sort_index(axis=1).astype({'深度':'float64','烈度':'float64','距离':'float64'})
df_result.head()


# In[42]:


df_result.astype({'深度':'float64','烈度':'float64','距离':'float64'},copy=False).dtypes


# In[43]:


df.equals(df_result)


# ## 第5章：练习一

# ### (a)

# In[44]:


df1 = pd.read_csv('data/Employee1.csv')
df2 = pd.read_csv('data/Employee2.csv')
df1.head()


# In[45]:


df2.head()


# In[46]:


L = list(set(df1['Name']).intersection(set(df2['Name'])))
L


# ### (b)

# In[47]:


df_b1 = df1[~df1['Name'].isin(L)]
df_b2 = df2[~df2['Name'].isin(L)]
df_b = pd.concat([df_b1,df_b2]).set_index('Name')
df_b.head()


# ### (c)

# In[48]:


df1 = pd.read_csv('data/Employee1.csv')
df2 = pd.read_csv('data/Employee2.csv')
df1['重复'] = ['Y_1' if df1.loc[i,'Name'] in L else 'N' for i in range(df1.shape[0])]
df2['重复'] = ['Y_2' if df2.loc[i,'Name'] in L else 'N' for i in range(df2.shape[0])]
df1 = df1.set_index(['Name','重复'])
df2 = df2.set_index(['Name','重复'])
df_c = pd.concat([df1,df2])
result = pd.DataFrame({'Company':[],'Name':[],'Age':[],'Height':[],'Weight':[],'Salary':[]})
group = df_c.groupby(['Company','重复'])
for i in L:
    first = group.get_group((i[0].upper(),'Y_1')).reset_index(level=1).loc[i,:][-4:]
    second = group.get_group((i[0].upper(),'Y_2')).reset_index(level=1).loc[i,:][-4:]
    mean = group.get_group((i[0].upper(),'N')).reset_index(level=1).mean()
    final = [i[0].upper(),i]
    for j in range(4):
        final.append(first[j] if abs(first[j]-mean[j])<abs(second[j]-mean[j]) else second[j])
    result = pd.concat([result,pd.DataFrame({result.columns.tolist()[k]:[final[k]] for k in range(6)})])
result = pd.concat([result.set_index('Name'),df_b])
for i in list('abcde'):
    for j in range(1,17):
        item = i+str(j)
        if item not in result.index:
            result = pd.concat([result,pd.DataFrame({'Company':[i.upper()],'Name':[item]
                 ,'Age':[np.nan],'Height':[np.nan],'Weight':[np.nan],'Salary':[np.nan]}).set_index('Name')])
result['Number'] = [int(i[1:]) for i in result.index]
result.reset_index().drop(columns='Name').set_index(['Company','Number']).sort_index()


# ## 第5章：练习二

# ### (a)

# In[49]:


df1 = pd.read_csv('data/Course1.csv')
df2 = pd.read_csv('data/Course2.csv')
df_a11,df_a12,df_a21,df_a22 =0,0,0,0
df_a11= df1.query('课程类别 in ["学科基础课","专业必修课","专业选修课"]')
df_a12= df1.query('课程类别 not in ["学科基础课","专业必修课","专业选修课"]')
df_a21= df2.query('课程类别 in ["学科基础课","专业必修课","专业选修课"]')
df_a22= df2.query('课程类别 not in ["学科基础课","专业必修课","专业选修课"]')
df_a11.head()


# ### (b)

# In[50]:


special = pd.concat([df_a11,df_a21])
common = pd.concat([df_a12,df_a22])
special.query('课程类别 not in ["学科基础课","专业必修课","专业选修课"]')


# In[51]:


common.query('课程类别 in ["学科基础课","专业必修课","专业选修课"]')


# ### (c)

# In[52]:


df = pd.concat([df1,df2])
special2 = df.query('课程类别 in ["学科基础课","专业必修课","专业选修课"]')
common2 = df.query('课程类别 not in ["学科基础课","专业必修课","专业选修课"]')
(special.equals(special2),common.equals(common2))


# ### (d)

# In[53]:


df['分数'] = df.groupby('课程类别').transform(lambda x: x.fillna(x.mean()))['分数']
df.isnull().all()


# In[54]:


special3 = df.query('课程类别 in ["学科基础课","专业必修课","专业选修课"]')
common3 = df.query('课程类别 not in ["学科基础课","专业必修课","专业选修课"]')
common3.head()


# ## 第6章：练习一
# ### (a)

# In[55]:


df = pd.read_csv('data/Missing_data_one.csv').convert_dtypes()
df.head()


# In[56]:


df.dtypes


# In[57]:


df[df['C'].isna()]


# ### (b)

# In[58]:


df = pd.read_csv('data/Missing_data_one.csv').convert_dtypes()
total_b = df['B'].sum()
min_b = df['B'].min()
df['A'] = pd.Series(list(zip(df['A'].values
                    ,df['B'].values))).apply(lambda x:x[0] if np.random.rand()>0.25*x[1]/min_b else np.nan)
df.head()


# ## 第6章：练习二

# ### (a)

# In[59]:


df = pd.read_csv('data/Missing_data_two.csv')
df.head()


# In[60]:


df.isna().sum()/df.shape[0]


# In[61]:


df_not2na = df[df.iloc[:,-3:].isna().sum(1)<=1]
df_not2na.head()


# ### (b)
# #### 分地区，用排序后的身高信息进行线性插值

# In[62]:


df_method_1 = df.copy()
for name,group in df_method_1.groupby('地区'):
    df_method_1.loc[group.index,'体重'] = group[['身高','体重']].sort_values(by='身高').interpolate()['体重']
df_method_1['体重'] = df_method_1['体重'].round(decimals=2)
df_method_1.head()


# ## 第7章：练习一
# ### (a)

# In[63]:


df = pd.read_csv('data/String_data_one.csv',index_col='人员编号').astype('str')
df.head()


# In[64]:


(df['姓名']+':'+df['国籍']+'国人，性别'
          +df['性别']+'，生于'
          +df['出生年']+'年'
          +df['出生月']+'月'+df['出生日']+'日').to_frame().rename(columns={0:'ID'}).head()


# ### (b)

# In[65]:


L_year = list('零一二三四五六七八九')
L_one = [s.strip() for s in list('  二三四五六七八九')]
L_two = [s.strip() for s in list(' 一二三四五六七八九')]
df_new = (df['姓名']+':'+df['国籍']+'国人，性别'+df['性别']+'，生于'
          +df['出生年'].str.replace(r'\d',lambda x:L_year[int(x.group(0))])+'年'
          +df['出生月'].apply(lambda x:x if len(x)==2 else '0'+x)\
                      .str.replace(r'(?P<one>[\d])(?P<two>\d?)',lambda x:L_one[int(x.group('one'))]
                      +bool(int(x.group('one')))*'十'+L_two[int(x.group('two'))])+'月'
          +df['出生日'].apply(lambda x:x if len(x)==2 else '0'+x)\
                      .str.replace(r'(?P<one>[\d])(?P<two>\d?)',lambda x:L_one[int(x.group('one'))]
                      +bool(int(x.group('one')))*'十'+L_two[int(x.group('two'))])+'日')\
          .to_frame().rename(columns={0:'ID'})
df_new.head()


# ### (c)

# In[66]:


dic_year = {i[0]:i[1] for i in zip(list('零一二三四五六七八九'),list('0123456789'))}
dic_two = {i[0]:i[1] for i in zip(list('十一二三四五六七八九'),list('0123456789'))}
dic_one = {'十':'1','二十':'2','三十':'3',None:''}
df_res = df_new['ID'].str.extract(r'(?P<姓名>[a-zA-Z]+):(?P<国籍>[\d])国人，性别(?P<性别>[\w])，生于(?P<出生年>[\w]{4})年(?P<出生月>[\w]+)月(?P<出生日>[\w]+)日')
df_res['出生年'] = df_res['出生年'].str.replace(r'(\w)+',lambda x:''.join([dic_year[x.group(0)[i]] for i in range(4)]))
df_res['出生月'] = df_res['出生月'].str.replace(r'(?P<one>\w?十)?(?P<two>[\w])',lambda x:dic_one[x.group('one')]+dic_two[x.group('two')]).str.replace(r'0','10')
df_res['出生日'] = df_res['出生日'].str.replace(r'(?P<one>\w?十)?(?P<two>[\w])',lambda x:dic_one[x.group('one')]+dic_two[x.group('two')]).str.replace(r'^0','10')
df_res.head()


# In[67]:


df_res.equals(df)


# ## 第7章：练习二
# #### 看题目可能会觉得疑惑，为什么会有(b)和(c)两道那么水的题，但是其中的坑只有做过/实际数据处理中遇到过才知道

# ### (a)

# In[68]:


df = pd.read_csv('data/String_data_two.csv')
df.head()


# In[69]:


df[df['col1'].str.contains(r'[北京]{2}|[上海]{2}')].head()


# ### (b)

# In[70]:


#df['col2'].mean() #报错
df['col2'][~(df['col2'].str.replace(r'-?\d+','True')=='True')] #这三行有问题


# In[71]:


df.loc[[309,396,485],'col2'] = [0,9,7]
df['col2'].astype('int').mean()


# ### (c)

# In[72]:


#df['col3'].mean() #报错
#df['col3'] #报错
df.columns


# In[73]:


df.columns = df.columns.str.strip()
df.columns


# In[74]:


df['col3'].head()


# In[75]:


#df['col3'].mean() #报错
df['col3'][~(df['col3'].str.replace(r'-?\d+\.?\d+','True')=='True')]


# In[76]:


df.loc[[28,122,332],'col3'] = [355.3567,9056.2253, 3534.6554]
df['col3'].astype('float').mean()


# ## 第8章 练习一

# ### (a)

# In[77]:


df = pd.read_csv('data/Earthquake.csv')
df.head()


# In[78]:


df_a = df.copy()
df_a['深度'] = pd.cut(df_a['深度'], [-1e-10,5,10,15,20,30,50,np.inf],labels=['Ⅰ','Ⅱ','Ⅲ','Ⅳ','Ⅴ','Ⅵ','Ⅶ'])
df_a.set_index('深度').sort_index().head()


# ### (b)

# In[79]:


df_a['烈度'] = pd.cut(df_a['烈度'], [-1e-10,3,4,5,np.inf],labels=['Ⅰ','Ⅱ','Ⅲ','Ⅳ'])
df_a.set_index(['深度','烈度']).sort_index().head()


# ## 第8章 练习二

# In[80]:


foo = pd.Categorical(['a', 'b'], categories=['a', 'b', 'c'])
bar = pd.Categorical(['d', 'e'], categories=['d', 'e', 'f'])
pd.crosstab(foo, bar)


# In[81]:


def my_crosstab(foo,bar):
    num = len(foo)
    s1 = pd.Series([i for i in list(foo.categories.union(set(foo)))],name='1nd var')
    s2 = [i for i in list(bar.categories.union(set(bar)))]
    df = pd.DataFrame({i:[0]*len(s1) for i in s2},index=s1)
    for i in range(num):
        df.at[foo[i],bar[i]] += 1
    return df.rename_axis('2st var',axis=1)
my_crosstab(foo,bar)


# ## 第9章：练习一

# In[82]:


df = pd.read_csv('data/time_series_one.csv', parse_dates=['日期'])
df.head()


# ### (a) 周日，注意dayofweek函数结果里0对应周一；6对应周日

# In[83]:


df['日期'].dt.dayofweek[df['销售额'].idxmax()]


# ### (b)

# In[84]:


holiday = pd.date_range(start='20170501', end='20170503').append(
          pd.date_range(start='20171001', end='20171007')).append(
          pd.date_range(start='20180215', end='20180221')).append(
          pd.date_range(start='20180501', end='20180503')).append(
          pd.date_range(start='20181001', end='20181007')).append(
          pd.date_range(start='20190204', end='20190224')).append(
          pd.date_range(start='20190501', end='20190503')).append(
          pd.date_range(start='20191001', end='20191007'))
result = df[~df['日期'].isin(holiday)].set_index('日期').resample('MS').sum()
result.head()


# ### (c)

# In[85]:


result = df[df['日期'].dt.dayofweek.isin([5,6])].set_index('日期').resample('QS').sum()
result.head()


# ### (d) 这里结果的日期是5天里的最后一天

# In[86]:


df_temp = df[~df['日期'].dt.dayofweek.isin([5,6])].set_index('日期').iloc[::-1]
L_temp,date_temp = [],[0]*df_temp.shape[0]
for i in range(df_temp.shape[0]//5):
    L_temp.extend([i]*5)
L_temp.extend([df_temp.shape[0]//5]*(df_temp.shape[0]-df_temp.shape[0]//5*5))
date_temp = pd.Series([i%5==0 for i in range(df_temp.shape[0])])
df_temp['num'] = L_temp
result = pd.DataFrame({'5天总额':df_temp.groupby('num')['销售额'].sum().values},
                       index=df_temp.reset_index()[date_temp]['日期']).iloc[::-1]
result.head()


# ### (e)

# In[87]:


from datetime import datetime 
df_temp = df.copy()
df_fri = df.shift(4)[df.shift(4)['日期'].dt.dayofweek==1]['销售额']
df_mon = df.shift(-4)[df.shift(-4)['日期'].dt.dayofweek==5]['销售额']
df_temp.loc[df_fri.index,['销售额']] = df_fri
df_temp.loc[df_mon.index,['销售额']] = df_mon
df_temp.loc[df_temp[df_temp['日期'].dt.year==2018]['日期'][
        df_temp[df_temp['日期'].dt.year==2018]['日期'].apply(
        lambda x:True if datetime.strptime(str(x).split()[0],'%Y-%m-%d').weekday() == 0 
        and 1 <= datetime.strptime(str(x).split()[0],'%Y-%m-%d').day <= 7 else False)].index,:]


# ## 第9章：练习二

# ### (a)

# In[88]:


df = pd.read_csv('data/time_series_one.csv',index_col='日期',parse_dates=['日期'])
df['销售额'].rolling(window=50,min_periods=1).mean().head()


# In[89]:


df['销售额'].rolling(window=50,min_periods=1).max().head()


# ### (b)

# In[90]:


def f(x):
    if len(x) == 6:
        return 1 if x[-1]>np.mean(x[:-1]) else 0
    else:
        return 0
result_b = df.loc[pd.date_range(start='20171227',end='20181231'),:].rolling(
                                                    window=6,min_periods=1).agg(f)[5:].head()
result_b.head()


# ### (c) 比较巧合，与(b)的结果一样

# In[91]:


def f(x):
    if len(x) == 8:
        return 1 if x[-1]>np.mean(x[:-1][pd.Series([
            False if i in [5,6] else True for i in x[:-1].index.dayofweek],index=x[:-1].index)]) else 0
    else:
        return 0
result_c = df.loc[pd.date_range(start='20171225',end='20181231'),:].rolling(
                                    window=8,min_periods=1).agg(f)[7:].head()
result_c.head()

