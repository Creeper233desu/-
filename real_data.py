"""
城市绿色物流配送调度优化系统
基于 K-means 聚类 + 最近邻贪心 + 2-opt 局部搜索
使用真实数据求解
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import math
import random
import os
from typing import List, Tuple, Dict, Optional

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ==================== 常量参数 ====================

STARTUP_COST = 400
EARLY_PENALTY = 20
LATE_PENALTY = 50
SERVICE_TIME = 20/60
CARBON_COST = 0.65

FUEL_PRICE = 7.61
ELEC_PRICE = 1.64

ETA_FUEL = 2.547
GAMMA_ELEC = 0.501

FUEL_LOAD_FACTOR = 0.4
ELEC_LOAD_FACTOR = 0.35

GREEN_ZONE_CENTER = (0, 0)
GREEN_ZONE_RADIUS = 10
GREEN_ZONE_START = 8.0
GREEN_ZONE_END = 16.0

DEPOT_X, DEPOT_Y = 20, 20


# ==================== 数据类定义 ====================

class Customer:
    """客户类"""
    def __init__(self, id, x, y, demand_weight, demand_volume, tw_start, tw_end):
        self.id = id
        self.x = x
        self.y = y
        self.demand_weight = demand_weight
        self.demand_volume = demand_volume
        self.tw_start = tw_start
        self.tw_end = tw_end

    def distance_to(self, other) -> float:
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)


class Depot:
    """配送中心"""
    def __init__(self, x=DEPOT_X, y=DEPOT_Y):
        self.id = 0
        self.x = x
        self.y = y

    def distance_to(self, other) -> float:
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)


class VehicleType:
    """车辆类型"""
    def __init__(self, name, max_weight, max_volume, count, energy_type):
        self.name = name
        self.max_weight = max_weight
        self.max_volume = max_volume
        self.count = count
        self.energy_type = energy_type


class Route:
    """路径类"""
    def __init__(self):
        self.customers: List[Customer] = []
        self.total_weight = 0.0
        self.total_volume = 0.0

    def can_add(self, customer: Customer, max_weight: float, max_volume: float) -> bool:
        return (self.total_weight + customer.demand_weight <= max_weight * 0.95 and
                self.total_volume + customer.demand_volume <= max_volume * 0.95)

    def add_customer(self, customer: Customer):
        self.customers.append(customer)
        self.total_weight += customer.demand_weight
        self.total_volume += customer.demand_volume

    @property
    def is_empty(self) -> bool:
        return len(self.customers) == 0


class CostBreakdown:
    """成本明细"""
    def __init__(self):
        self.startup = 0.0
        self.fuel = 0.0
        self.electric = 0.0
        self.carbon_cost = 0.0
        self.early_penalty = 0.0
        self.late_penalty = 0.0
        self.total = 0.0
        self.carbon_kg = 0.0
        self.distance_km = 0.0
        self.vehicle_count = 0
        self.customers_served = 0


# ==================== 真实数据读取 ====================

def load_real_data(excel_dir: str = ".") -> Tuple[List[Customer], Depot, List[VehicleType], np.ndarray]:
    """从Excel文件读取真实数据"""
    
    orders_path = os.path.join(excel_dir, "订单信息.xlsx")
    distance_path = os.path.join(excel_dir, "距离矩阵.xlsx")
    coords_path = os.path.join(excel_dir, "客户坐标信息.xlsx")
    timewindow_path = os.path.join(excel_dir, "时间窗.xlsx")
    
    print("  正在读取真实数据文件...")
    
    # 1. 读取订单信息，聚合到客户
    print(f"    - 读取订单信息: {orders_path}")
    orders_df = pd.read_excel(orders_path)
    orders_df['重量'] = orders_df['重量'].fillna(0)
    orders_df['体积'] = orders_df['体积'].fillna(0)
    
    customer_demands = orders_df.groupby('目标客户编号').agg({
        '重量': 'sum',
        '体积': 'sum'
    }).reset_index()
    customer_demands.columns = ['客户编号', '总重量', '总体积']
    
    print(f"    - 订单数: {len(orders_df)}, 客户数: {len(customer_demands)}")
    
    # 2. 读取距离矩阵
    print(f"    - 读取距离矩阵: {distance_path}")
    dist_df = pd.read_excel(distance_path, index_col=0)
    if '客户' in dist_df.columns:
        dist_df = dist_df.drop(columns=['客户'])
    distance_matrix = dist_df.values.astype(float)
    print(f"    - 距离矩阵维度: {distance_matrix.shape}")
    
    # 3. 读取客户坐标
    print(f"    - 读取客户坐标: {coords_path}")
    coords_df = pd.read_excel(coords_path)
    
    depot_row = coords_df[coords_df['类型'] == '配送中心'].iloc[0]
    depot = Depot(depot_row['X (km)'], depot_row['Y (km)'])
    print(f"    - 配送中心坐标: ({depot.x}, {depot.y})")
    
    # 4. 读取时间窗
    print(f"    - 读取时间窗: {timewindow_path}")
    tw_df = pd.read_excel(timewindow_path)
    
    def time_to_hours(time_str):
        if pd.isna(time_str):
            return 0.0
        parts = str(time_str).split(':')
        return float(parts[0]) + float(parts[1]) / 60.0
    
    tw_df['开始时间_h'] = tw_df['开始时间'].apply(time_to_hours)
    tw_df['结束时间_h'] = tw_df['结束时间'].apply(time_to_hours)
    
    # 5. 创建Customer对象
    customers = []
    customer_coords = coords_df[coords_df['类型'] == '客户']
    
    for _, coord_row in customer_coords.iterrows():
        cid = int(coord_row['ID'])
        x = coord_row['X (km)']
        y = coord_row['Y (km)']
        
        demand_row = customer_demands[customer_demands['客户编号'] == cid]
        if len(demand_row) > 0:
            weight = demand_row['总重量'].values[0]
            volume = demand_row['总体积'].values[0]
        else:
            weight = 0
            volume = 0
        
        tw_row = tw_df[tw_df['客户编号'] == cid]
        if len(tw_row) > 0:
            tw_start = tw_row['开始时间_h'].values[0]
            tw_end = tw_row['结束时间_h'].values[0]
        else:
            tw_start = 8.0
            tw_end = 20.0
        
        customers.append(Customer(cid, x, y, weight, volume, tw_start, tw_end))
    
    total_weight = sum(c.demand_weight for c in customers)
    total_volume = sum(c.demand_volume for c in customers)
    print(f"    - 总需求: 重量={total_weight:.0f}kg, 体积={total_volume:.1f}m³")
    
    # 6. 车辆类型
    vehicle_types = [
        VehicleType('FV1 (燃油,3t)', 3000, 13.5, 60, 'fuel'),
        VehicleType('FV2 (燃油,1.5t)', 1500, 10.8, 50, 'fuel'),
        VehicleType('FV3 (燃油,1.25t)', 1250, 6.5, 50, 'fuel'),
        VehicleType('EV1 (电动,3t)', 3000, 15.0, 10, 'electric'),
        VehicleType('EV2 (电动,1.25t)', 1250, 8.5, 15, 'electric'),
    ]
    
    return customers, depot, vehicle_types, distance_matrix


# ==================== 速度与时间 ====================

def get_speed_by_time(hour: float) -> float:
    hour = hour % 24
    if (9.0 <= hour < 10.0) or (13.0 <= hour < 15.0):
        return 55.3
    elif (10.0 <= hour < 11.5) or (15.0 <= hour < 17.0):
        return 35.4
    else:
        return 9.8


def get_time_period_name(hour: float) -> str:
    hour = hour % 24
    if (9.0 <= hour < 10.0) or (13.0 <= hour < 15.0):
        return '顺畅'
    elif (10.0 <= hour < 11.5) or (15.0 <= hour < 17.0):
        return '一般'
    else:
        return '拥堵'


# ==================== 能耗计算 ====================

def calculate_fpk(speed: float) -> float:
    return 0.0025 * speed**2 - 0.2554 * speed + 31.75


def calculate_epk(speed: float) -> float:
    return 0.0014 * speed**2 - 0.12 * speed + 36.19


def calculate_fuel_cost(distance_km: float, speed: float, load_ratio: float) -> Tuple[float, float]:
    fpk = calculate_fpk(speed)
    load_factor = 1 + FUEL_LOAD_FACTOR * load_ratio
    fuel_L = fpk * distance_km / 100 * load_factor
    cost = fuel_L * FUEL_PRICE
    carbon = fuel_L * ETA_FUEL
    return cost, carbon


def calculate_electric_cost(distance_km: float, speed: float, load_ratio: float) -> Tuple[float, float]:
    epk = calculate_epk(speed)
    load_factor = 1 + ELEC_LOAD_FACTOR * load_ratio
    elec_kwh = epk * distance_km / 100 * load_factor
    cost = elec_kwh * ELEC_PRICE
    carbon = elec_kwh * GAMMA_ELEC
    return cost, carbon


# ==================== K-Means 聚类（加速版） ====================

def kmeans_cluster(customers: List[Customer], n_clusters: int, depot: Depot,
                   max_iterations: int = 20) -> Tuple[Dict[int, List[Customer]], np.ndarray]:
    """对客户进行K-means聚类（加速版）"""
    np.random.seed(42)
    coords = np.array([[c.x, c.y] for c in customers])
    
    n_clusters = min(n_clusters, len(customers))
    center_indices = np.random.choice(len(customers), n_clusters, replace=False)
    centers = coords[center_indices].copy()
    
    for iteration in range(max_iterations):
        # 分配点到最近的中心
        clusters = {i: [] for i in range(n_clusters)}
        for i, coord in enumerate(coords):
            distances = np.sum((centers - coord)**2, axis=1)  # 用平方距离，更快
            nearest = np.argmin(distances)
            clusters[nearest].append(customers[i])
        
        # 更新中心
        new_centers = np.zeros_like(centers)
        for i in range(n_clusters):
            if clusters[i]:
                cluster_coords = np.array([[c.x, c.y] for c in clusters[i]])
                new_centers[i] = cluster_coords.mean(axis=0)
            else:
                new_centers[i] = centers[i]
        
        if np.allclose(centers, new_centers, rtol=1e-4):
            break
        centers = new_centers
    
    return clusters, centers


def nearest_neighbor_route(customers: List[Customer], depot: Depot,
                           vehicle_type: VehicleType) -> List[Route]:
    """使用最近邻算法构建路径（修复版）"""
    routes = []
    unvisited = list(customers)
    
    while unvisited:
        route = Route()
        current_pos = depot
        visited_this_route = []
        
        # 先找第一个能装下的客户
        first_customer = None
        first_dist = float('inf')
        for customer in unvisited:
            if customer.demand_weight <= vehicle_type.max_weight and \
               customer.demand_volume <= vehicle_type.max_volume:
                dist = current_pos.distance_to(customer)
                if dist < first_dist:
                    first_dist = dist
                    first_customer = customer
        
        if first_customer is None:
            # 有客户超过单车容量，可能需求太大
            print(f"      警告: 簇内有客户超过单车容量({vehicle_type.max_weight}kg)")
            # 把超容量的客户单独装一辆车
            for customer in unvisited[:]:
                if customer.demand_weight > vehicle_type.max_weight:
                    # 用最大车单独服务
                    big_route = Route()
                    big_route.add_customer(customer)
                    routes.append(big_route)
                    unvisited.remove(customer)
                    print(f"      客户{customer.id}单独一车: {customer.demand_weight:.0f}kg")
            continue
        
        route.add_customer(first_customer)
        current_pos = first_customer
        unvisited.remove(first_customer)
        
        # 继续添加其他客户
        while True:
            best_customer = None
            best_distance = float('inf')
            
            for customer in unvisited:
                if route.can_add(customer, vehicle_type.max_weight, vehicle_type.max_volume):
                    dist = current_pos.distance_to(customer)
                    if dist < best_distance:
                        best_distance = dist
                        best_customer = customer
            
            if best_customer is None:
                break
            
            route.add_customer(best_customer)
            current_pos = best_customer
            unvisited.remove(best_customer)
        
        if not route.is_empty:
            routes.append(route)
    
    return routes


# ==================== 2-opt 局部优化（加速版） ====================

def two_opt_improve(route: List[Customer], depot: Depot, max_iter: int = 15) -> List[Customer]:
    """对路径进行2-opt优化（加速版）"""
    if len(route) <= 3:
        return route
    
    def calc_route_distance(r: List[Customer]) -> float:
        total = depot.distance_to(r[0])
        for i in range(len(r) - 1):
            total += r[i].distance_to(r[i+1])
        total += r[-1].distance_to(depot)
        return total
    
    best_route = route.copy()
    best_dist = calc_route_distance(best_route)
    
    for iteration in range(max_iter):
        improved = False
        for i in range(len(best_route) - 2):
            for j in range(i + 2, len(best_route)):
                new_route = best_route[:i+1] + best_route[i+1:j+1][::-1] + best_route[j+1:]
                new_dist = calc_route_distance(new_route)
                
                if new_dist < best_dist - 0.01:
                    best_route = new_route
                    best_dist = new_dist
                    improved = True
                    break
            if improved:
                break
        
        if not improved:
            break
    
    return best_route


# ==================== 成本计算 ====================

def calculate_route_cost_detailed(route: Route, vehicle_type: VehicleType,
                                   depot: Depot, start_time: float = 8.0) -> CostBreakdown:
    cost = CostBreakdown()
    
    current_time = start_time
    current_pos = depot
    
    for customer in route.customers:
        dist = current_pos.distance_to(customer)
        speed = get_speed_by_time(current_time)
        
        load_ratio = route.total_weight / vehicle_type.max_weight
        if vehicle_type.energy_type == 'fuel':
            energy_cost, carbon = calculate_fuel_cost(dist, speed, load_ratio)
            cost.fuel += energy_cost
        else:
            energy_cost, carbon = calculate_electric_cost(dist, speed, load_ratio)
            cost.electric += energy_cost
        
        cost.total += energy_cost
        cost.carbon_cost += carbon * CARBON_COST
        cost.total += carbon * CARBON_COST
        cost.carbon_kg += carbon
        cost.distance_km += dist
        
        current_time += dist / max(1, speed)
        
        if current_time < customer.tw_start:
            wait = customer.tw_start - current_time
            cost.early_penalty += wait * EARLY_PENALTY
            cost.total += wait * EARLY_PENALTY
            current_time = customer.tw_start
        elif current_time > customer.tw_end:
            delay = current_time - customer.tw_end
            cost.late_penalty += delay * LATE_PENALTY
            cost.total += delay * LATE_PENALTY
        
        current_time += SERVICE_TIME
        current_pos = customer
    
    # 返回配送中心
    dist_back = current_pos.distance_to(depot)
    speed = get_speed_by_time(current_time)
    if vehicle_type.energy_type == 'fuel':
        energy_cost, carbon = calculate_fuel_cost(dist_back, speed, 0)
        cost.fuel += energy_cost
    else:
        energy_cost, carbon = calculate_electric_cost(dist_back, speed, 0)
        cost.electric += energy_cost
    
    cost.total += energy_cost + carbon * CARBON_COST
    cost.carbon_cost += carbon * CARBON_COST
    cost.carbon_kg += carbon
    cost.distance_km += dist_back
    
    cost.startup = STARTUP_COST
    cost.total += STARTUP_COST
    cost.vehicle_count = 1
    cost.customers_served = len(route.customers)
    
    return cost


# ==================== 主求解器 ====================

class SimpleVRPSolver:
    """简化VRP求解器"""
    
    def __init__(self, customers: List[Customer], depot: Depot,
                 vehicle_types: List[VehicleType],
                 distance_matrix: Optional[np.ndarray] = None):
        self.customers = customers
        self.depot = depot
        self.vehicle_types = vehicle_types
        self.distance_matrix = distance_matrix
    
    def solve(self, n_clusters: Optional[int] = None,
          prefer_electric: bool = False,
          max_electric_vehicles: int = 999) -> Tuple[List[Route], Dict[str, int], float, CostBreakdown]:
        """求解VRP"""
        if n_clusters is None:
            total_weight = sum(c.demand_weight for c in self.customers)
            n_clusters = int(total_weight / 3000 * 1.1)
        
        n_clusters = min(n_clusters, len(self.customers), 40)
        
        print(f"    聚类数: {n_clusters} (客户数: {len(self.customers)})")
        
        clusters, _ = kmeans_cluster(self.customers, n_clusters, self.depot)
        print(f"    聚类完成，开始路径规划...")
        
        all_routes = []
        vehicle_usage = {}
        overall_cost = CostBreakdown()
        
        # 统计可用的电车数量
        ev1_available = 10
        ev2_available = 15
        fuel_used = 0
        electric_used = 0
        
        # 按簇总重量降序排列，大簇优先用电车
        sorted_clusters = sorted(clusters.items(), 
                                key=lambda x: sum(c.demand_weight for c in x[1]), 
                                reverse=True)
        
        for cluster_id, cluster_customers in sorted_clusters:
            if not cluster_customers:
                continue
            
            total_w = sum(c.demand_weight for c in cluster_customers)
            total_v = sum(c.demand_volume for c in cluster_customers)
            
            # ===== 智能选择车辆类型 =====
            best_vt = None
            
            if prefer_electric:
                # 问题2绿色区：强制用电车
                available = [vt for vt in self.vehicle_types if vt.energy_type == 'electric']
            else:
                # 问题1：优先用电车，用完再用油车
                if ev1_available > 0 and total_w <= 3000 and total_v <= 15.0:
                    # 用EV1
                    best_vt = next(vt for vt in self.vehicle_types if vt.name.startswith('EV1'))
                    ev1_available -= 1
                elif ev2_available > 0 and total_w <= 1250 and total_v <= 8.5:
                    # 用EV2
                    best_vt = next(vt for vt in self.vehicle_types if vt.name.startswith('EV2'))
                    ev2_available -= 1
            
            # 如果没有分配电车，用油车
            if best_vt is None:
                # 选最合适的燃油车
                for vt in self.vehicle_types:
                    if vt.energy_type == 'fuel' and vt.max_weight >= total_w * 0.85:
                        best_vt = vt
                        break
                if best_vt is None:
                    best_vt = self.vehicle_types[0]  # 最大的燃油车
                fuel_used += 1
            else:
                electric_used += 1
            
            vehicle_usage[best_vt.name] = vehicle_usage.get(best_vt.name, 0) + 1
            
            # 路径构建
            routes = nearest_neighbor_route(cluster_customers, self.depot, best_vt)
            
            # 2-opt优化
            for route in routes:
                route.customers = two_opt_improve(route.customers, self.depot)
            
            all_routes.extend(routes)
        
        print(f"    电车使用: {electric_used}辆, 油车使用: {fuel_used}辆")
        print(f"    路径规划完成，计算成本...")
        
        # 成本计算
        for route in all_routes:
            if not route.customers:
                continue
            # 匹配合适的车辆类型
            for vt in self.vehicle_types:
                if vt.max_weight >= route.total_weight:
                    cost_detail = calculate_route_cost_detailed(route, vt, self.depot)
                    break
            else:
                cost_detail = calculate_route_cost_detailed(route, self.vehicle_types[0], self.depot)
            
            overall_cost.startup += cost_detail.startup
            overall_cost.fuel += cost_detail.fuel
            overall_cost.electric += cost_detail.electric
            overall_cost.carbon_cost += cost_detail.carbon_cost
            overall_cost.early_penalty += cost_detail.early_penalty
            overall_cost.late_penalty += cost_detail.late_penalty
            overall_cost.total += cost_detail.total
            overall_cost.carbon_kg += cost_detail.carbon_kg
            overall_cost.distance_km += cost_detail.distance_km
            overall_cost.vehicle_count += 1
            overall_cost.customers_served += cost_detail.customers_served
        
        print(f"    成本计算完成!")
        
        return all_routes, vehicle_usage, overall_cost.total, overall_cost

# ==================== 绿色配送区 ====================

def is_in_green_zone(x: float, y: float) -> bool:
    return math.sqrt(x**2 + y**2) <= GREEN_ZONE_RADIUS


def filter_customers_by_zone(customers: List[Customer]) -> Tuple[List[Customer], List[Customer]]:
    in_zone = [c for c in customers if is_in_green_zone(c.x, c.y)]
    out_zone = [c for c in customers if not is_in_green_zone(c.x, c.y)]
    return in_zone, out_zone


# ==================== 可视化 ====================

def plot_speed_distribution():
    """绘制速度分布图"""
    hours = np.linspace(8, 17, 500)
    speeds = [get_speed_by_time(h) for h in hours]
    
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(hours, speeds, 'b-', linewidth=3)
    ax.fill_between(hours, 0, speeds, color='blue', alpha=0.1)
    ax.axvspan(8, 16, color='green', alpha=0.05)
    ax.text(12, 3, '绿色配送区限行时段 (8:00-16:00)', ha='center', fontsize=10, color='darkgreen')
    ax.set_xlabel('时间 (小时)', fontsize=12)
    ax.set_ylabel('平均速度 (km/h)', fontsize=12)
    ax.set_title('全天车速变化（时变特性）', fontsize=14, fontweight='bold')
    ax.set_xlim(8, 17)
    ax.set_ylim(0, 70)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_energy_curve():
    """绘制能耗U型曲线"""
    speeds = np.linspace(5, 90, 200)
    fpk = [calculate_fpk(v) for v in speeds]
    epk = [calculate_epk(v) for v in speeds]
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(speeds, fpk, 'r-', linewidth=2, label='燃油车 FPK (L/100km)')
    ax.plot(speeds, epk, 'b-', linewidth=2, label='新能源车 EPK (kWh/100km)')
    ax.set_xlabel('速度 (km/h)', fontsize=12)
    ax.set_ylabel('能耗', fontsize=12)
    ax.set_title('能耗-速度关系曲线（U型）', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def visualize_solution(customers: List[Customer], depot: Depot, routes: List[Route],
                       title: str = "配送方案"):
    """可视化配送方案"""
    fig, ax = plt.subplots(figsize=(14, 10))
    
    green_circle = plt.Circle(GREEN_ZONE_CENTER, GREEN_ZONE_RADIUS,
                              fill=True, alpha=0.1, color='green', label='绿色配送区')
    ax.add_patch(green_circle)
    
    ax.scatter(depot.x, depot.y, c='red', s=300, marker='*',
               label='配送中心', zorder=5, edgecolors='darkred')
    
    in_zone_x, in_zone_y = [], []
    out_zone_x, out_zone_y = [], []
    for c in customers:
        if is_in_green_zone(c.x, c.y):
            in_zone_x.append(c.x); in_zone_y.append(c.y)
        else:
            out_zone_x.append(c.x); out_zone_y.append(c.y)
    
    ax.scatter(in_zone_x, in_zone_y, c='darkgreen', s=40, alpha=0.6,
               marker='s', label=f'绿色区内客户 ({len(in_zone_x)}个)')
    ax.scatter(out_zone_x, out_zone_y, c='gray', s=30, alpha=0.4,
               marker='o', label=f'绿色区外客户 ({len(out_zone_x)}个)')
    
    colors = plt.cm.tab20(np.linspace(0, 1, max(len(routes), 1)))
    for i, route in enumerate(routes):
        if not route.customers:
            continue
        path_x = [depot.x] + [c.x for c in route.customers] + [depot.x]
        path_y = [depot.y] + [c.y for c in route.customers] + [depot.y]
        ax.plot(path_x, path_y, '-', color=colors[i], linewidth=1.5, alpha=0.7, marker='o', markersize=4)
    
    ax.set_xlabel('X (km)', fontsize=12)
    ax.set_ylabel('Y (km)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.show()


def plot_cost_pie_chart(cost: CostBreakdown, title: str = "成本构成"):
    """绘制成本构成饼图"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    categories = ['启动成本', '燃油费用', '电费', '碳排放成本', '早到等待', '晚到惩罚']
    values = [cost.startup, cost.fuel, cost.electric, cost.carbon_cost,
              cost.early_penalty, cost.late_penalty]
    
    filtered_cats, filtered_vals = [], []
    for cat, val in zip(categories, values):
        if val > 0.1:
            filtered_cats.append(cat)
            filtered_vals.append(val)
    
    colors_pie = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']
    filtered_colors = [colors_pie[i] for i, v in enumerate(values) if v > 0.1]
    
    wedges, texts, autotexts = ax1.pie(
        filtered_vals, labels=filtered_cats, colors=filtered_colors,
        autopct='%1.1f%%', startangle=90, textprops={'fontsize': 10}
    )
    for at in autotexts:
        at.set_fontweight('bold')
    ax1.set_title('成本构成比例', fontsize=13, fontweight='bold')
    
    bars = ax2.barh(filtered_cats, filtered_vals, color=filtered_colors, edgecolor='black')
    ax2.set_xlabel('金额 (元)', fontsize=11)
    ax2.set_title('各项成本金额', fontsize=13, fontweight='bold')
    for bar, val in zip(bars, filtered_vals):
        ax2.text(bar.get_width() + max(filtered_vals) * 0.02, bar.get_y() + bar.get_height()/2,
                f'{val:.0f} 元', va='center', fontsize=10, fontweight='bold')
    
    fig.suptitle(title, fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()


def plot_vehicle_usage_bar(vehicle_usage: Dict[str, int], title: str = "车辆使用情况"):
    """绘制车辆使用柱状图"""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    names = list(vehicle_usage.keys())
    counts = list(vehicle_usage.values())
    
    colors_list = ['#E74C3C' if 'FV' in n else '#2ECC71' for n in names]
    bars = ax.bar(names, counts, color=colors_list, edgecolor='black')
    
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                str(count), ha='center', fontsize=12, fontweight='bold')
    
    ax.set_xlabel('车辆类型', fontsize=11)
    ax.set_ylabel('使用数量', fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    fuel_patch = mpatches.Patch(color='#E74C3C', label='燃油车')
    elec_patch = mpatches.Patch(color='#2ECC71', label='新能源车')
    ax.legend(handles=[fuel_patch, elec_patch], fontsize=10)
    plt.tight_layout()
    plt.show()


def plot_comparison_bar(cost1: CostBreakdown, cost2: CostBreakdown,
                        label1: str = "问题1", label2: str = "问题2"):
    """绘制对比柱状图"""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    categories = ['启动成本', '燃油费用', '电费', '碳排放成本', '早到等待', '晚到惩罚']
    values1 = [cost1.startup, cost1.fuel, cost1.electric, cost1.carbon_cost,
               cost1.early_penalty, cost1.late_penalty]
    values2 = [cost2.startup, cost2.fuel, cost2.electric, cost2.carbon_cost,
               cost2.early_penalty, cost2.late_penalty]
    
    x = np.arange(len(categories))
    width = 0.35
    
    axes[0].bar(x - width/2, values1, width, label=label1, color='#3498DB', edgecolor='black')
    axes[0].bar(x + width/2, values2, width, label=label2, color='#E74C3C', edgecolor='black')
    axes[0].set_ylabel('成本 (元)', fontsize=11)
    axes[0].set_title('各项成本对比', fontsize=12, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(categories, rotation=30, ha='right', fontsize=9)
    axes[0].legend(fontsize=9)
    axes[0].grid(axis='y', alpha=0.3)
    
    totals = [cost1.total, cost2.total]
    bars_total = axes[1].bar([label1, label2], totals, color=['#3498DB', '#E74C3C'],
                             edgecolor='black', width=0.5)
    for bar, total in zip(bars_total, totals):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(totals)*0.02,
                    f'{total:.0f} 元', ha='center', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('总成本 (元)', fontsize=11)
    axes[1].set_title('总成本对比', fontsize=12, fontweight='bold')
    axes[1].grid(axis='y', alpha=0.3)
    
    carbon_values = [cost1.carbon_kg, cost2.carbon_kg]
    bars_carbon = axes[2].bar([label1, label2], carbon_values, color=['#2ECC71', '#E67E22'],
                              edgecolor='black', width=0.5)
    for bar, cv in zip(bars_carbon, carbon_values):
        axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(carbon_values)*0.02,
                    f'{cv:.1f} kg', ha='center', fontsize=12, fontweight='bold')
    axes[2].set_ylabel('碳排放 (kg)', fontsize=11)
    axes[2].set_title('碳排放对比', fontsize=12, fontweight='bold')
    axes[2].grid(axis='y', alpha=0.3)
    
    fig.suptitle('环保政策影响对比分析', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


# ==================== 主程序 ====================

def main():
    print("=" * 70)
    print("     城市绿色物流配送调度优化系统")
    print("     基于 K-means + 最近邻贪心 + 2-opt 局部搜索")
    print("=" * 70)
    
    # 读取数据
    print("\n[步骤1] 读取真实数据...")
    
    # 假设Excel文件在当前目录，如果不在请修改路径
    # 使用脚本文件所在的目录
    data_dir = os.path.dirname(os.path.abspath(__file__))
    
    if os.path.exists(os.path.join(data_dir, "订单信息.xlsx")):
        customers, depot, vehicle_types, distance_matrix = load_real_data(data_dir)
        print("\n  ✅ 成功读取真实数据！")
    else:
        print("\n  ⚠️ 未找到数据文件，请将Excel文件放在程序同目录下！")
        return
    
    in_zone, out_zone = filter_customers_by_zone(customers)
    
    print(f"\n  数据概览:")
    print(f"  客户总数: {len(customers)}")
    print(f"  绿色配送区内客户: {len(in_zone)}")
    print(f"  绿色配送区外客户: {len(out_zone)}")
    print(f"  配送中心坐标: ({depot.x}, {depot.y})")
    
    # 数据可视化
    print("\n[步骤2] 数据可视化...")
    plot_speed_distribution()
    plot_energy_curve()
    
    # ============ 问题1 ============
    print("\n" + "=" * 70)
    print("问题1: 静态环境下的车辆调度（无政策限制）")
    print("=" * 70)
    
    solver = SimpleVRPSolver(customers, depot, vehicle_types, distance_matrix)
    routes_q1, usage_q1, total_q1, cost_q1 = solver.solve()
    
    print(f"\n  结果汇总:")
    print(f"  ├─ 使用车辆数: {len(routes_q1)}")
    print(f"  ├─ 车辆类型: {usage_q1}")
    print(f"  ├─ 总成本: {cost_q1.total:.2f} 元")
    print(f"  ├─ 碳排放: {cost_q1.carbon_kg:.2f} kg")
    print(f"  └─ 行驶总里程: {cost_q1.distance_km:.2f} km")
    
    print(f"\n  成本明细:")
    print(f"  ├─ 启动成本:     {cost_q1.startup:>10.2f} 元")
    print(f"  ├─ 燃油费用:     {cost_q1.fuel:>10.2f} 元")
    print(f"  ├─ 电费:         {cost_q1.electric:>10.2f} 元")
    print(f"  ├─ 碳排放成本:   {cost_q1.carbon_cost:>10.2f} 元")
    print(f"  ├─ 早到等待:     {cost_q1.early_penalty:>10.2f} 元")
    print(f"  └─ 晚到惩罚:     {cost_q1.late_penalty:>10.2f} 元")
    
    visualize_solution(customers, depot, routes_q1, "问题1: 静态无限制配送方案")
    plot_cost_pie_chart(cost_q1, "问题1: 配送成本构成")
    plot_vehicle_usage_bar(usage_q1, "问题1: 车辆类型使用情况")
    
    # ============ 问题2 ============
    print("\n" + "=" * 70)
    print("问题2: 绿色配送区限行政策 (8:00-16:00 禁止燃油车进入)")
    print("=" * 70)
    
    green_vehicles = [vt for vt in vehicle_types if vt.energy_type == 'electric']
    
    if in_zone:
        green_solver = SimpleVRPSolver(in_zone, depot, green_vehicles, distance_matrix)
        green_routes, green_usage, _, green_cost = green_solver.solve(
            n_clusters=max(2, len(in_zone)//15), prefer_electric=True)
    else:
        green_routes, green_usage, green_cost = [], {}, CostBreakdown()
    
    if out_zone:
        normal_solver = SimpleVRPSolver(out_zone, depot, vehicle_types, distance_matrix)
        normal_routes, normal_usage, _, normal_cost = normal_solver.solve()
    else:
        normal_routes, normal_usage, normal_cost = [], {}, CostBreakdown()
    
    all_routes_q2 = green_routes + normal_routes
    
    cost_q2 = CostBreakdown()
    for attr in ['startup', 'fuel', 'electric', 'carbon_cost', 'early_penalty', 
                 'late_penalty', 'total', 'carbon_kg', 'distance_km']:
        setattr(cost_q2, attr, getattr(green_cost, attr) + getattr(normal_cost, attr))
    cost_q2.vehicle_count = len(all_routes_q2)
    cost_q2.customers_served = len(customers)
    
    usage_q2 = {}
    for u in [green_usage, normal_usage]:
        for k, v in u.items():
            usage_q2[k] = usage_q2.get(k, 0) + v
    
    print(f"\n  结果汇总:")
    print(f"  ├─ 绿色区内新能源车: {len(green_routes)} 辆")
    print(f"  ├─ 绿色区外车辆:     {len(normal_routes)} 辆")
    print(f"  ├─ 总车辆数:         {len(all_routes_q2)}")
    print(f"  ├─ 总成本:           {cost_q2.total:.2f} 元")
    print(f"  └─ 碳排放:           {cost_q2.carbon_kg:.2f} kg")
    
    visualize_solution(customers, depot, all_routes_q2, "问题2: 绿色配送区限行方案")
    plot_cost_pie_chart(cost_q2, "问题2: 配送成本构成")
    plot_vehicle_usage_bar(usage_q2, "问题2: 车辆类型使用情况")
    
    # 对比分析
    print(f"\n  {'='*60}")
    print(f"  政策影响对比分析")
    print(f"  {'='*60}")
    print(f"  {'指标':<20} {'问题1':>12} {'问题2':>12} {'变化':>12}")
    print(f"  {'-'*60}")
    print(f"  {'总成本(元)':<20} {cost_q1.total:>12.2f} {cost_q2.total:>12.2f} {cost_q2.total-cost_q1.total:>+12.2f}")
    print(f"  {'碳排放(kg)':<20} {cost_q1.carbon_kg:>12.2f} {cost_q2.carbon_kg:>12.2f} {cost_q2.carbon_kg-cost_q1.carbon_kg:>+12.2f}")
    print(f"  {'车辆数':<20} {len(routes_q1):>12} {len(all_routes_q2):>12} {len(all_routes_q2)-len(routes_q1):>+12}")
    change_rate = (cost_q2.total - cost_q1.total) / cost_q1.total * 100
    carbon_reduction = (cost_q1.carbon_kg - cost_q2.carbon_kg) / cost_q1.carbon_kg * 100
    print(f"  {'成本变化率':<20} {'':>12} {'':>12} {change_rate:>+11.2f}%")
    print(f"  {'碳减排率':<20} {'':>12} {'':>12} {carbon_reduction:>+11.2f}%")
    
    plot_comparison_bar(cost_q1, cost_q2, "问题1(无限行)", "问题2(有限行)")
    
    # ============ 问题3 ============
    print("\n" + "=" * 70)
    print("问题3: 动态事件响应策略")
    print("=" * 70)
    
    print("""
    ┌─────────────────────────────────────────────────────────────┐
    │                    动态调度策略设计                           │
    ├─────────────────────────────────────────────────────────────┤
    │  1. 新增订单处理 (插入启发式)                                 │
    │     ├─ 评估所有车辆剩余容量                                   │
    │     ├─ 寻找最佳插入位置 (最小额外距离)                         │
    │     └─ 无法插入则派遣新车                                     │
    │                                                             │
    │  2. 订单取消处理                                             │
    │     ├─ 从路径中移除客户                                       │
    │     └─ 检查相邻路径是否可合并                                 │
    │                                                             │
    │  3. 地址变更处理                                             │
    │     └─ 取消旧订单 + 新增订单                                   │
    │                                                             │
    │  4. 时间窗调整处理                                           │
    │     ├─ 检查当前到达时间是否满足新时间窗                        │
    │     └─ 不满足则重新插入                                       │
    │                                                             │
    │  5. 实时更新机制                                             │
    │     ├─ 滚动时域优化 (每15分钟检查)                             │
    │     └─ 优先保持已出发车辆的路线稳定                           │
    └─────────────────────────────────────────────────────────────┘
    """)
    
    print("=" * 70)
    print("求解完成！")
    print("=" * 70)


if __name__ == "__main__":
    main()