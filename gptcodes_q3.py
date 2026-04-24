# =========================================
# Problem 3 动态调度
# =========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import copy

# 中文
rcParams['font.sans-serif']=['SimHei']
rcParams['axes.unicode_minus']=False

#########################################
# 参数
#########################################
START_COST=400
FUEL_COST=7.61
ETA=2.547
CARBON_PRICE=0.65

#########################################
# 数据
#########################################
orders=pd.read_excel('订单信息.xlsx')
dist_df=pd.read_excel('距离矩阵.xlsx')
coords=pd.read_excel('客户坐标信息.xlsx')

orders.columns=orders.columns.str.strip()
coords.columns=coords.columns.str.strip()

#########################################
# 需求聚合
#########################################

demand=(
orders.groupby('目标客户编号')
.agg(weight=('重量','sum'))
.reset_index()
)

demand=demand[
demand.weight>0
]

customers=demand['目标客户编号'].tolist()

#########################################
# 距离
#########################################
if '客户' in dist_df.columns:
    dist_df=dist_df.drop(columns=['客户'])

D=dist_df.values

#########################################
# 坐标列
#########################################
xcol=[c for c in coords.columns if 'X' in c][0]
ycol=[c for c in coords.columns if 'Y' in c][0]

#########################################
# 能耗函数
#########################################
def fuel(v):
    return 0.0025*v*v-0.2554*v+31.75

#########################################
# 初始静态方案（直接用Savings）
#########################################

def savings(customers,Q=3000):

    routes=[[0,c,0] for c in customers]

    S=[]

    for i in customers:
        for j in customers:
            if i<j:
                s=D[0,i]+D[0,j]-D[i,j]
                S.append((s,i,j))

    S.sort(reverse=True)

    for _,i,j in S:

        ri=rj=None

        for r in routes:
            if i in r:
                ri=r
            if j in r:
                rj=r

        if ri!=rj and ri and rj:

            li=demand[
             demand['目标客户编号'].isin(ri)
            ]['weight'].sum()

            lj=demand[
             demand['目标客户编号'].isin(rj)
            ]['weight'].sum()

            if li+lj<=Q:

                if ri[-2]==i and rj[1]==j:

                    new=ri[:-1]+rj[1:]
                    routes.remove(ri)
                    routes.remove(rj)
                    routes.append(new)

    return routes

initial_routes=savings(customers)

#########################################
# 路线成本
#########################################

def route_cost(route):

    c=START_COST

    for i in range(len(route)-1):

        d=D[
        route[i],
        route[i+1]
        ]

        e=(d/100)*fuel(35)

        c+=FUEL_COST*e
        c+=CARBON_PRICE*(ETA*e)

    return c


def total_cost(routes):

    return sum(
    route_cost(r)
    for r in routes
    )

base_cost=total_cost(initial_routes)

#########################################
# 插入算法（新增订单）
#########################################

def insert_customer(routes,new_customer):

    best_delta=1e20
    best_route=None
    best_pos=None

    for rid,r in enumerate(routes):

        for p in range(1,len(r)):

            i=r[p-1]
            j=r[p]

            delta=(
            D[i,new_customer]+
            D[new_customer,j]-
            D[i,j]
            )

            if delta<best_delta:
                best_delta=delta
                best_route=rid
                best_pos=p

    new_routes=copy.deepcopy(routes)

    new_routes[
    best_route
    ].insert(best_pos,new_customer)

    return new_routes

#########################################
# 删除订单
#########################################

def remove_customer(routes,cust):

    new=[]

    for r in routes:

        if cust in r:

            rr=[x for x in r if x!=cust]

            if len(rr)>=3:
                new.append(rr)
        else:
            new.append(r)

    return new

#########################################
# 时间窗变化（简单交换修复）
#########################################

def adjust_route(routes,cust):

    new=copy.deepcopy(routes)

    for r in new:

        if cust in r:

            idx=r.index(cust)

            if idx>1:
                r[idx],r[idx-1]=r[idx-1],r[idx]

    return new

#########################################
# 动态事件模拟
#########################################

# 事件1 新增订单（虚拟客户）
# 假设新增客户90
new_customer=90

routes1=insert_customer(
initial_routes,
new_customer
)

cost1=total_cost(routes1)

# 事件2 订单取消
routes2=remove_customer(
routes1,
12
)

cost2=total_cost(routes2)

# 事件3 时间窗调整
routes3=adjust_route(
routes2,
27
)

cost3=total_cost(routes3)

#########################################
# 输出结果
#########################################

print('初始成本=',round(base_cost,2))
print('事件1后成本=',round(cost1,2))
print('事件2后成本=',round(cost2,2))
print('事件3后成本=',round(cost3,2))

print('扰动成本=',round(cost3-base_cost,2))

#########################################
# 图1 动态事件成本变化
#########################################

plt.figure()

plt.plot(
['初始','新增订单','取消订单','时间窗变化'],
[base_cost,cost1,cost2,cost3],
marker='o'
)

plt.title('动态事件下成本变化')
plt.ylabel('总成本')
plt.show()

#########################################
# 图2 动态重规划路径图
#########################################

plt.figure(figsize=(10,10))

plt.scatter(
coords[xcol],
coords[ycol]
)

for r in routes3:

    for i in range(len(r)-1):

        a=coords.iloc[r[i]]
        b=coords.iloc[r[i+1]]

        plt.plot(
        [a[xcol],b[xcol]],
        [a[ycol],b[ycol]]
        )

plt.title('动态事件后的实时调度路径')
plt.show()

#########################################
# 图3 扰动来源分析
#########################################

extra1=cost1-base_cost
extra2=cost2-cost1
extra3=cost3-cost2

plt.figure()

plt.bar(
['新增订单',
'取消订单',
'时间窗调整'],
[extra1,
extra2,
extra3]
)

plt.title('各类动态事件扰动影响')
plt.show()

#########################################
# 导出结果
#########################################

result=[]

for i,r in enumerate(routes3):

    result.append([
    i+1,
    str(r),
    route_cost(r)
    ])

out=pd.DataFrame(
result,
columns=[
'车辆',
'动态路径',
'成本'
]
)

out.to_excel(
'第三问动态调度结果.xlsx',
index=False
)

print('结果已保存 第三问动态调度结果.xlsx')
