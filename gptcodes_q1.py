import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from copy import deepcopy

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']  # 设置中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示异常

#########################################
# 1 读取数据
#########################################

orders=pd.read_excel('订单信息.xlsx')
dist_df=pd.read_excel('距离矩阵.xlsx')
coords=pd.read_excel('客户坐标信息.xlsx')
tw=pd.read_excel('时间窗.xlsx')

# 去掉列名空格
orders.columns=orders.columns.str.strip()
coords.columns=coords.columns.str.strip()
tw.columns=tw.columns.str.strip()

#########################################
# 2 聚合需求并删除0需求客户
#########################################

# 根据你的表头若不同自己改字段名
customer_demand=(
orders.groupby('目标客户编号')
.agg(weight=('重量','sum'),
     volume=('体积','sum'))
.reset_index()
)

# 删除0需求客户
customer_demand=customer_demand[
    (customer_demand['weight']>0) |
    (customer_demand['volume']>0)
].copy()

print('有效客户数量:',len(customer_demand))

# 有效客户编号
valid_customers=customer_demand['目标客户编号'].tolist()

# 加入配送中心0号
nodes=[0]+valid_customers

#########################################
# 3 提取有效距离矩阵
#########################################

if '客户' in dist_df.columns:
    dist_df=dist_df.drop(columns=['客户'])

D_full=dist_df.values

# 仅保留有效节点
D=D_full[np.ix_(nodes,nodes)]

# 建立映射
node_index={node:i for i,node in enumerate(nodes)}

#########################################
# 4 时间窗
#########################################

# 假设字段名如下，若不一致改一下
# earliest latest

try:
    tw_valid=tw[tw['客户编号'].isin(valid_customers)]
except:
    tw_valid=tw.copy()

#########################################
# 5 车辆参数
#########################################

vehicle={
    'Q':3000,
    'V':13.5,
    'fixed_cost':400
}

#########################################
# 6 速度模型
#########################################

def sample_speed(t):

    if 8<=t<9 or 11.5<=t<13:
        v=np.random.normal(9.8,4.7)

    elif 10<=t<11.5 or 15<=t<17:
        v=np.random.normal(35.4,5.2)

    else:
        v=np.random.normal(55.3,0.1)

    return max(v,5)


def fuel(v):
    return 0.0025*v*v-0.2554*v+31.75

#########################################
# 7 Clarke-Wright Savings
#########################################

def savings_algorithm():

    routes=[]

    # 初始每客户单独一路
    for c in valid_customers:
        routes.append([0,c,0])

    savings=[]

    for i in valid_customers:
        for j in valid_customers:
            if i<j:
                ii=node_index[i]
                jj=node_index[j]

                s=D[0,ii]+D[0,jj]-D[ii,jj]
                savings.append((s,i,j))

    savings.sort(reverse=True)

    for s,i,j in savings:

        ri=None
        rj=None

        for r in routes:
            if i in r:
                ri=r
            if j in r:
                rj=r

        if ri!=rj and ri is not None and rj is not None:

            # 仅简单容量约束（重量）
            load_i=customer_demand[
                customer_demand['目标客户编号'].isin(ri)
            ]['weight'].sum()

            load_j=customer_demand[
                customer_demand['目标客户编号'].isin(rj)
            ]['weight'].sum()

            if load_i+load_j<=vehicle['Q']:

                if ri[-2]==i and rj[1]==j:
                    new_route=ri[:-1]+rj[1:]
                    routes.remove(ri)
                    routes.remove(rj)
                    routes.append(new_route)

    return routes

#########################################
# 8 2-opt优化
#########################################

def route_distance(route):

    total=0

    for i in range(len(route)-1):
        a=node_index[route[i]]
        b=node_index[route[i+1]]
        total+=D[a,b]

    return total


def two_opt(route):

    best=route
    improved=True

    while improved:

        improved=False

        for i in range(1,len(best)-2):
            for j in range(i+1,len(best)-1):

                new=best[:]
                new[i:j]=reversed(best[i:j])

                if route_distance(new)<route_distance(best):
                    best=new
                    improved=True

    return best

#########################################
# 9 成本计算
#########################################

def route_cost(route):

    cost=vehicle['fixed_cost']

    t=8

    for k in range(len(route)-1):

        i=node_index[route[k]]
        j=node_index[route[k+1]]

        d=D[i,j]

        v=sample_speed(t)

        travel=d/v

        t+=travel

        load_ratio=0.7

        energy=(d/100)*fuel(v)*(1+0.4*load_ratio)

        fuel_cost=7.61*energy

        carbon_cost=0.65*(2.547*energy)

        cost+=fuel_cost+carbon_cost

        # 简单软时间窗惩罚（示意）
        if t>17:
            cost+=50*(t-17)

    return cost

#########################################
# 10 求解
#########################################

routes=savings_algorithm()

print('初始车辆数:',len(routes))

optimized=[]

total_cost=0

for r in routes:

    rr=two_opt(r)

    optimized.append(rr)

    c=route_cost(rr)

    total_cost+=c

print('总成本=',round(total_cost,2))

#########################################
# 11 输出路径
#########################################

for i,r in enumerate(optimized):
    print('车辆',i+1,':',r)

#########################################
# 12 可视化
#########################################

# 自动识别坐标列
possible_x=[c for c in coords.columns if 'X' in c or 'x' in c]
possible_y=[c for c in coords.columns if 'Y' in c or 'y' in c]

xcol=possible_x[0]
ycol=possible_y[0]

plt.figure(figsize=(10,10))

plt.scatter(
    coords[xcol],
    coords[ycol],
    s=40
)

for r in optimized:

    for i in range(len(r)-1):

        a=coords.iloc[r[i]]
        b=coords.iloc[r[i+1]]

        plt.plot(
            [a[xcol],b[xcol]],
            [a[ycol],b[ycol]]
        )

plt.title('车辆路径')
plt.show()

#########################################
# 13 成本分解（论文可直接用）
#########################################

startup=len(optimized)*400

print('启动成本:',startup)
print('平均单车成本:',total_cost/len(optimized))

#########################################
# 图1 成本构成饼图
#########################################

startup=len(optimized)*400

energy_cost=0
carbon_cost=0

for r in optimized:

    for k in range(len(r)-1):

        i=node_index[r[k]]
        j=node_index[r[k+1]]

        d=D[i,j]

        v=35

        e=(d/100)*fuel(v)*(1+0.4*0.7)

        energy_cost+=7.61*e
        carbon_cost+=0.65*(2.547*e)

penalty=max(
    total_cost-
    startup-
    energy_cost-
    carbon_cost,
0
)

plt.figure(figsize=(8,8))

plt.pie(
    [startup,energy_cost,carbon_cost,penalty],
    labels=[
        '启动成本',
        '能源成本',
        '碳排放成本',
        '时间窗惩罚'
    ],
    autopct='%1.1f%%'
)

plt.title('Cost Composition')
plt.show()

#########################################
# 图2 需求分布
#########################################

plt.figure(figsize=(9,6))

plt.hist(
 customer_demand['weight'],
 bins=15
)

plt.xlabel('需求重量 (kg)')
plt.ylabel('客户数量')
plt.title('需求分布')
plt.show()

#########################################
# 图3 车辆载重利用率
#########################################

loads=[]

for r in optimized:

    load=customer_demand[
      customer_demand['目标客户编号'].isin(r)
    ]['weight'].sum()

    loads.append(load/vehicle['Q'])

plt.figure(figsize=(10,6))

plt.bar(
 range(1,len(loads)+1),
 loads
)

plt.axhline(
 np.mean(loads),
 linestyle='--'
)

plt.xlabel('车辆编号')
plt.ylabel('载重利用率')
plt.title('车辆载重利用率')

plt.show()