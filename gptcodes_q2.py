# ==========================================
# 华中杯A题 第二问完整独立代码（修正版）
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# ---------- 中文显示 ----------
rcParams['font.sans-serif']=['SimHei']
rcParams['axes.unicode_minus']=False

# ---------- 参数 ----------
START_COST=400
FUEL_COST=7.61
ELEC_COST=1.64
CARBON_PRICE=0.65
ETA=2.547
GAMMA=0.501

BASELINE_COST=25483
BASELINE_EMISSION=6200

# 车辆参数
FUEL_Q=3000
EV_Q=1250

# ---------------------------
# 数据读取
# ---------------------------
orders=pd.read_excel('订单信息.xlsx')
dist_df=pd.read_excel('距离矩阵.xlsx')
coords=pd.read_excel('客户坐标信息.xlsx')

orders.columns=orders.columns.str.strip()
coords.columns=coords.columns.str.strip()

# ---------------------------
# 聚合并剔除0需求客户
# ---------------------------

demand=(
orders.groupby('目标客户编号')
.agg(weight=('重量','sum'),
volume=('体积','sum'))
.reset_index()
)

demand=demand[
(demand.weight>0)|
(demand.volume>0)
]

customers=demand['目标客户编号'].tolist()

print('有效客户数:',len(customers))

# ---------------------------
# 距离矩阵
# ---------------------------
if '客户' in dist_df.columns:
    dist_df=dist_df.drop(columns=['客户'])

D=dist_df.values

# ---------------------------
# 坐标列自动识别
# ---------------------------
xcol=[c for c in coords.columns if 'X' in c or 'x' in c][0]
ycol=[c for c in coords.columns if 'Y' in c or 'y' in c][0]

# ---------------------------
# 绿色配送区识别
# ---------------------------
coords['green']=(
coords[xcol]**2+
coords[ycol]**2
)<=100

green=[]
outside=[]

for c in customers:

    if coords.iloc[c]['green']:
        green.append(c)
    else:
        outside.append(c)

print('绿色区客户:',len(green))
print('区外客户:',len(outside))

# ---------------------------
# 能耗函数
# ---------------------------
def fuel(v):
    return 0.0025*v*v-0.2554*v+31.75

def electric(v):
    return 0.001*v*v-0.1*v+36.194

# ---------------------------
# Savings算法
# ---------------------------

def savings(customers,Q):

    routes=[[0,c,0] for c in customers]

    S=[]

    for i in customers:
        for j in customers:
            if i<j:
                s=D[0,i]+D[0,j]-D[i,j]
                S.append((s,i,j))

    S.sort(reverse=True)

    for _,i,j in S:

        ri=None
        rj=None

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

# ---------------------------
# 全局重新规划
# ---------------------------

# 绿色区由电车
EV_routes=savings(green,EV_Q)

# 区外燃油车
Fuel_routes=savings(outside,FUEL_Q)

all_routes=[]
vehicle_type=[]

for r in Fuel_routes:
    all_routes.append(r)
    vehicle_type.append('燃油车')

for r in EV_routes:
    all_routes.append(r)
    vehicle_type.append('新能源车')

print('总车辆数:',len(all_routes))

# ---------------------------
# 成本计算
# ---------------------------

def route_cost(route,is_ev=False):

    cost=START_COST

    for i in range(len(route)-1):

        d=D[
            route[i],
            route[i+1]
        ]

        v=35

        if not is_ev:

            e=(d/100)*fuel(v)

            cost+=FUEL_COST*e
            cost+=CARBON_PRICE*(ETA*e)

        else:

            e=(d/100)*electric(v)

            cost+=ELEC_COST*e
            cost+=CARBON_PRICE*(GAMMA*e)

    return cost

# ---------------------------
# 成本分解
# ---------------------------

startup_cost=len(all_routes)*400
energy_cost=0
carbon_cost=0
total_emission=0
total_cost=0

for k,r in enumerate(all_routes):

    ev=(vehicle_type[k]=='新能源车')

    total_cost+=route_cost(r,ev)

    for i in range(len(r)-1):

        d=D[r[i],r[i+1]]

        if not ev:

            e=(d/100)*fuel(35)
            energy_cost+=FUEL_COST*e
            carbon_cost+=CARBON_PRICE*(ETA*e)
            total_emission+=ETA*e

        else:

            e=(d/100)*electric(35)
            energy_cost+=ELEC_COST*e
            carbon_cost+=CARBON_PRICE*(GAMMA*e)
            total_emission+=GAMMA*e

print()
print('总成本=',round(total_cost,2))
print('总碳排放=',round(total_emission,2))

# ---------------------------
# 输出全部路径（完整结果）
# ---------------------------

for i,r in enumerate(all_routes):

    print(
    vehicle_type[i],
    i+1,
    r,
    '成本=',round(
      route_cost(
      r,
      vehicle_type[i]=='新能源车'
      ),2)
    )

# ====================================
# 图1 全部路线图（关键）
# ====================================

plt.figure(figsize=(10,10))

plt.scatter(
coords[xcol],
coords[ycol],
label='客户点'
)

circle=plt.Circle(
(0,0),10,
fill=False,
linestyle='--'
)

plt.gca().add_patch(circle)

for k,r in enumerate(all_routes):

    for i in range(len(r)-1):

        a=coords.iloc[r[i]]
        b=coords.iloc[r[i+1]]

        if vehicle_type[k]=='燃油车':
            style='-'
        else:
            style='--'

        plt.plot(
        [a[xcol],b[xcol]],
        [a[ycol],b[ycol]],
        linestyle=style
        )

plt.title('政策下全部配送路径规划')
plt.show()

# ====================================
# 图2 成本构成
# ====================================

plt.figure(figsize=(8,8))

plt.pie(
[
startup_cost,
energy_cost,
carbon_cost
],
labels=[
'启动成本',
'能源成本',
'碳成本'
],
autopct='%1.1f%%'
)

plt.title('总成本构成')
plt.show()

# ====================================
# 图3 车辆结构变化
# ====================================

fuel_num=sum(
1 for x in vehicle_type
if x=='燃油车'
)

ev_num=sum(
1 for x in vehicle_type
if x=='新能源车'
)

plt.figure()

plt.bar(
['燃油车','新能源车'],
[fuel_num,ev_num]
)

plt.title('政策下车辆结构')
plt.ylabel('车辆数')
plt.show()

# ====================================
# 图4 政策前后比较
# ====================================

plt.figure(figsize=(8,6))

x=np.arange(2)

plt.bar(
x-0.15,
[BASELINE_COST,total_cost],
0.3,
label='总成本'
)

plt.bar(
x+0.15,
[
BASELINE_EMISSION,
total_emission
],
0.3,
label='碳排放'
)

plt.xticks(
x,
['政策前','政策后']
)

plt.legend()

plt.title('政策影响对比')
plt.show()

# ====================================
# 导出结果表
# ====================================

result=[]

for i,r in enumerate(all_routes):

    result.append([
    vehicle_type[i],
    i+1,
    str(r),
    route_cost(
    r,
    vehicle_type[i]=='新能源车'
    )
    ])

out=pd.DataFrame(
result,
columns=[
'车型',
'车辆编号',
'路径',
'成本'
]
)

out.to_excel(
'第二问调度结果.xlsx',
index=False
)

print('\n结果保存完成：第二问调度结果.xlsx')
