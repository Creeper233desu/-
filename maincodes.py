import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
import math
import random
# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei','Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
# ==================== 数据类定义 ====================

class Customer:
    """客户类"""
    def __init__(self, id, x, y, demand_weight, demand_volume, tw_start, tw_end):
        self.id = id
        self.x = x
        self.y = y
        self.demand_weight = demand_weight
        self.demand_volume = demand_volume
        self.tw_start = tw_start  # 时间窗开始（小时，如9.0表示9:00）
        self.tw_end = tw_end      # 时间窗结束
        
    def distance_to(self, other) -> float:
        """计算到另一点的距离"""
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)

class Depot:
    """配送中心"""
    def __init__(self, x=20, y=20):
        self.id = 0
        self.x = x
        self.y = y
    
    def distance_to(self, other) -> float:
        """计算到另一点的距离"""
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)

class VehicleType:
    """车辆类型"""
    def __init__(self, name, max_weight, max_volume, count, energy_type):
        self.name = name
        self.max_weight = max_weight
        self.max_volume = max_volume
        self.count = count
        self.energy_type = energy_type  # 'fuel' 或 'electric'

class Route:
    """路径类"""
    def __init__(self):
        self.customers = []  # 不包含配送中心
        self.total_weight = 0
        self.total_volume = 0
    
    def can_add(self, customer: Customer, max_weight: float, max_volume: float) -> bool:
        """判断能否添加客户"""
        return (self.total_weight + customer.demand_weight <= max_weight and
                self.total_volume + customer.demand_volume <= max_volume)
    
    def add_customer(self, customer: Customer):
        """添加客户"""
        self.customers.append(customer)
        self.total_weight += customer.demand_weight
        self.total_volume += customer.demand_volume
    
    @property
    def is_empty(self) -> bool:
        return len(self.customers) == 0

# ==================== 速度与时间计算 ====================

def get_speed_by_time(hour: float) -> float:
    """根据时间获取车速（取均值）"""
    hour = hour % 24
    if (9 <= hour < 10) or (13 <= hour < 15):
        return 55.3  # 顺畅时段
    elif (10 <= hour < 11.5) or (15 <= hour < 17):
        return 35.4  # 一般时段
    else:
        return 9.8   # 拥堵时段

def travel_time(distance_km: float, departure_hour: float) -> float:
    """计算旅行时间（小时）"""
    speed = get_speed_by_time(departure_hour)
    return distance_km / max(1, speed)

# ==================== 能耗计算 ====================

def calculate_fuel_cost(distance_km: float, speed: float, load_ratio: float) -> Tuple[float, float]:
    """计算燃油成本 (元) 和 碳排放 (kg)"""
    # 百公里油耗 (L/100km)
    fpk = 0.0025 * speed**2 - 0.2554 * speed + 31.75
    # 载荷修正系数
    load_factor = 1 + 0.4 * load_ratio
    # 实际油耗 (L)
    fuel_L = fpk * distance_km / 100 * load_factor
    # 成本
    cost = fuel_L * 7.61
    # 碳排放
    carbon = fuel_L * 2.547
    return cost, carbon

def calculate_electric_cost(distance_km: float, speed: float, load_ratio: float) -> Tuple[float, float]:
    """计算电耗成本 (元) 和 碳排放 (kg)"""
    # 百公里电耗 (kWh/100km)
    epk = 0.0014 * speed**2 - 0.12 * speed + 36.19
    # 载荷修正系数
    load_factor = 1 + 0.35 * load_ratio
    # 实际电耗 (kWh)
    elec_kwh = epk * distance_km / 100 * load_factor
    # 成本
    cost = elec_kwh * 1.64
    # 碳排放
    carbon = elec_kwh * 0.501
    return cost, carbon

# ==================== K-Means 聚类 ====================

def kmeans_cluster(customers: List[Customer], n_clusters: int, depot: Depot,
                   max_iterations: int = 100) -> Dict[int, List[Customer]]:
    """
    对客户进行K-means聚类
    返回: {簇编号: [Customer列表]}
    """
    # 初始化聚类中心（随机选择）
    np.random.seed(42)
    coords = np.array([[c.x, c.y] for c in customers])
    
    # 随机选择初始中心点
    center_indices = np.random.choice(len(customers), n_clusters, replace=False)
    centers = coords[center_indices].copy()
    
    for _ in range(max_iterations):
        # 分配每个点到最近的中心
        clusters = {i: [] for i in range(n_clusters)}
        for i, coord in enumerate(coords):
            distances = [np.linalg.norm(coord - center) for center in centers]
            nearest = np.argmin(distances)
            clusters[nearest].append(customers[i])
        
        # 更新聚类中心
        new_centers = np.zeros_like(centers)
        for i in range(n_clusters):
            if clusters[i]:
                cluster_coords = np.array([[c.x, c.y] for c in clusters[i]])
                new_centers[i] = cluster_coords.mean(axis=0)
            else:
                new_centers[i] = centers[i]
        
        # 检查收敛
        if np.allclose(centers, new_centers):
            break
        centers = new_centers
    
    return clusters

# ==================== 最近邻贪心路径构建 ====================

def nearest_neighbor_route(customers: List[Customer], depot: Depot,
                           vehicle_type: VehicleType) -> List[Route]:
    """
    使用最近邻算法构建路径
    """
    routes = []
    unvisited = list(customers)  # 未访问的客户
    
    while unvisited:
        route = Route()
        current_pos = depot  # 当前位置
        
        while unvisited:
            # 找到最近的可服务客户
            best_customer = None
            best_distance = float('inf')
            
            for customer in unvisited:
                if route.can_add(customer, vehicle_type.max_weight, vehicle_type.max_volume):
                    dist = current_pos.distance_to(customer)
                    if dist < best_distance:
                        best_distance = dist
                        best_customer = customer
            
            if best_customer is None:
                break  # 不能添加更多客户了
            
            route.add_customer(best_customer)
            current_pos = best_customer
            unvisited.remove(best_customer)
        
        if not route.is_empty:
            routes.append(route)
    
    return routes

# ==================== 2-opt 局部优化 ====================

def two_opt_improve(route: List[Customer], depot: Depot) -> List[Customer]:
    """
    对路径进行2-opt优化，减少总距离
    """
    if len(route) <= 2:
        return route
    
    improved = True
    best_route = route.copy()
    
    while improved:
        improved = False
        
        for i in range(len(best_route) - 1):
            for j in range(i + 2, len(best_route)):
                # 计算当前距离
                old_dist = _segment_distance(best_route, i, j, depot)
                # 计算翻转后的距离
                new_route = best_route[:i+1] + best_route[i+1:j+1][::-1] + best_route[j+1:]
                new_dist = _segment_distance(new_route, i, j, depot)
                
                if new_dist < old_dist:
                    best_route = new_route
                    improved = True
                    break
            if improved:
                break
    
    return best_route

def _segment_distance(route: List[Customer], i: int, j: int, depot: Depot) -> float:
    """计算路径中某段的距离"""
    # 前一个点
    prev = depot if i == -1 else route[i]
    # 后一个点
    next_point = depot if j + 1 >= len(route) else route[j + 1]
    
    return prev.distance_to(route[i + 1]) + route[j].distance_to(next_point)

# ==================== 路径成本计算 ====================

def calculate_route_cost(route: Route, vehicle_type: VehicleType, depot: Depot,
                         start_time: float = 8.0) -> Dict:
    """
    计算单条路径的详细成本
    """
    STARTUP_COST = 400
    EARLY_PENALTY = 20  # 元/小时
    LATE_PENALTY = 50   # 元/小时
    SERVICE_TIME = 20/60  # 20分钟
    CARBON_COST = 0.65   # 元/kg CO2
    
    total_cost = STARTUP_COST
    total_carbon = 0
    current_time = start_time
    current_pos = depot
    
    for customer in route.customers:
        # 行驶距离和时间
        dist = current_pos.distance_to(customer)
        speed = get_speed_by_time(current_time)
        t_time = dist / max(1, speed)
        
        # 能耗成本
        load_ratio = route.total_weight / vehicle_type.max_weight
        if vehicle_type.energy_type == 'fuel':
            energy_cost, carbon = calculate_fuel_cost(dist, speed, load_ratio)
        else:
            energy_cost, carbon = calculate_electric_cost(dist, speed, load_ratio)
        
        total_cost += energy_cost
        total_cost += carbon * CARBON_COST
        total_carbon += carbon
        
        # 更新时间
        current_time += t_time
        
        # 时间窗惩罚
        if current_time < customer.tw_start:
            wait_hours = customer.tw_start - current_time
            total_cost += wait_hours * EARLY_PENALTY
            current_time = customer.tw_start
        elif current_time > customer.tw_end:
            delay_hours = current_time - customer.tw_end
            total_cost += delay_hours * LATE_PENALTY
        
        # 服务时间
        current_time += SERVICE_TIME
        current_pos = customer
    
    # 返回配送中心
    dist_back = current_pos.distance_to(depot)
    speed = get_speed_by_time(current_time)
    if vehicle_type.energy_type == 'fuel':
        energy_cost, carbon = calculate_fuel_cost(dist_back, speed, 0)
    else:
        energy_cost, carbon = calculate_electric_cost(dist_back, speed, 0)
    
    total_cost += energy_cost
    total_cost += carbon * CARBON_COST
    total_carbon += carbon
    
    return {
        'total_cost': total_cost,
        'carbon': total_carbon,
        'vehicle_type': vehicle_type.name
    }

# ==================== 主求解器 ====================

class SimpleVRPSolver:
    """简化VRP求解器"""
    
    def __init__(self, customers: List[Customer], depot: Depot, 
                 vehicle_types: List[VehicleType]):
        self.customers = customers
        self.depot = depot
        self.vehicle_types = vehicle_types
    
    def solve(self, n_clusters: int = None) -> Tuple[List[Route], str, float]:
        """
        求解VRP
        返回: (路线列表, 使用的车辆类型, 总成本)
        """
        # 确定聚类数量
        if n_clusters is None:
            # 估算需要的车辆数
            total_weight = sum(c.demand_weight for c in self.customers)
            avg_capacity = np.mean([vt.max_weight for vt in self.vehicle_types])
            n_clusters = max(3, int(total_weight / avg_capacity * 1.2))
        
        print(f"  聚类数: {n_clusters}")
        
        # 步骤1: K-means聚类
        print("  步骤1: K-means聚类...")
        clusters = kmeans_cluster(self.customers, n_clusters, self.depot)
        
        # 步骤2: 为每个簇选择车辆类型并规划路径
        print("  步骤2: 贪心路径规划...")
        all_routes = []
        vehicle_usage = {}
        
        for cluster_id, cluster_customers in clusters.items():
            if not cluster_customers:
                continue
            
            # 计算簇的总需求
            total_w = sum(c.demand_weight for c in cluster_customers)
            
            # 选择合适的车辆类型
            best_vt = None
            for vt in sorted(self.vehicle_types, key=lambda x: x.max_weight):
                if vt.max_weight >= total_w * 0.8:  # 80%的利用率
                    best_vt = vt
                    break
            if best_vt is None:
                best_vt = self.vehicle_types[0]  # 使用最大车辆
            
            vehicle_usage[best_vt.name] = vehicle_usage.get(best_vt.name, 0) + 1
            
            # 最近邻构建路径
            routes = nearest_neighbor_route(cluster_customers, self.depot, best_vt)
            all_routes.extend(routes)
        
        # 步骤3: 2-opt优化
        print("  步骤3: 2-opt局部优化...")
        for route in all_routes:
            route.customers = two_opt_improve(route.customers, self.depot)
        
        # 计算总成本
        total_cost = 0
        for route in all_routes:
            # 为每辆车找到对应的类型
            for vt_name, _ in vehicle_usage.items():
                vt = next(vt for vt in self.vehicle_types if vt.name == vt_name)
                cost_info = calculate_route_cost(route, vt, self.depot)
                total_cost += cost_info['total_cost']
        
        return all_routes, vehicle_usage, total_cost

# ==================== 绿色配送区处理 ====================

def is_in_green_zone(x: float, y: float, center: Tuple[float, float] = (0, 0), 
                     radius: float = 10) -> bool:
    """判断是否在绿色配送区内"""
    return math.sqrt((x - center[0])**2 + (y - center[1])**2) <= radius

def filter_green_zone_customers(customers: List[Customer]) -> Tuple[List[Customer], List[Customer]]:
    """分离绿色配送区内外的客户"""
    in_zone = []
    out_zone = []
    for c in customers:
        if is_in_green_zone(c.x, c.y):
            in_zone.append(c)
        else:
            out_zone.append(c)
    return in_zone, out_zone

# ==================== 可视化 ====================

def visualize_solution(customers: List[Customer], depot: Depot, routes: List[Route],
                       title: str = "配送方案"):
    """可视化配送方案"""
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # 绿色配送区
    green_circle = plt.Circle((0, 0), 10, fill=True, alpha=0.15, 
                               color='green', label='绿色配送区')
    ax.add_patch(green_circle)
    
    # 配送中心
    ax.scatter(depot.x, depot.y, c='red', s=300, marker='*', 
               label='配送中心', zorder=5, edgecolors='darkred')
    
    # 所有客户
    xs = [c.x for c in customers]
    ys = [c.y for c in customers]
    ax.scatter(xs, ys, c='gray', s=30, alpha=0.5, label='客户点')
    
    # 路线
    colors = plt.cm.tab20(np.linspace(0, 1, len(routes)))
    for i, route in enumerate(routes):
        if not route.customers:
            continue
        
        # 构建完整路径
        path_x = [depot.x] + [c.x for c in route.customers] + [depot.x]
        path_y = [depot.y] + [c.y for c in route.customers] + [depot.y]
        
        ax.plot(path_x, path_y, '-o', color=colors[i], linewidth=1.5,
                markersize=5, alpha=0.8, label=f'路线{i+1}')
    
    ax.set_xlabel('X (km)', fontsize=12)
    ax.set_ylabel('Y (km)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), 
              fontsize=8, ncol=2, borderaxespad=0, framealpha=0.8)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    ax.set_xlim(-40, 40)
    ax.set_ylim(-40, 40)
    
    plt.tight_layout()
    plt.show()

# ==================== 数据生成（示例） ====================

def generate_sample_data(n_customers: int = 98) -> Tuple[List[Customer], Depot, List[VehicleType]]:
    """生成示例数据"""
    np.random.seed(42)
    random.seed(42)
    
    # 配送中心
    depot = Depot(20, 20)
    
    # 生成客户
    customers = []
    for i in range(1, n_customers + 1):
        # 随机位置（偏向于有一个中心聚类）
        if random.random() < 0.3:
            # 绿色配送区内
            angle = random.uniform(0, 2 * math.pi)
            r = random.uniform(0, 10)
            x = r * math.cos(angle)
            y = r * math.sin(angle)
        else:
            x = random.uniform(-35, 35)
            y = random.uniform(-35, 35)
        
        weight = random.uniform(50, 500)  # kg
        volume = random.uniform(0.1, 3.0)  # m³
        
        # 时间窗
        tw_start = random.uniform(8, 17)
        tw_end = tw_start + random.uniform(1, 3)
        
        customers.append(Customer(i, x, y, weight, volume, tw_start, tw_end))
    
    # 车辆类型
    vehicle_types = [
        VehicleType('FV1', 3000, 13.5, 60, 'fuel'),
        VehicleType('FV2', 1500, 10.8, 50, 'fuel'),
        VehicleType('FV3', 1250, 6.5, 50, 'fuel'),
        VehicleType('EV1', 3000, 15.0, 10, 'electric'),
        VehicleType('EV2', 1250, 8.5, 15, 'electric'),
    ]
    
    return customers, depot, vehicle_types

# ==================== 主程序 ====================

def main():
    print("=" * 60)
    print("城市绿色物流配送调度 - 简化求解方案")
    print("=" * 60)
    
    # 生成数据
    print("\n[1] 生成示例数据...")
    customers, depot, vehicle_types = generate_sample_data(98)
    print(f"    客户数: {len(customers)}")
    print(f"    车辆类型: {len(vehicle_types)} 种")
    
    # 统计绿色配送区
    in_zone, out_zone = filter_green_zone_customers(customers)
    print(f"    绿色配送区内客户: {len(in_zone)}")
    print(f"    绿色配送区外客户: {len(out_zone)}")
    
    # ============ 问题1: 无限制配送 ============
    print("\n" + "=" * 60)
    print("问题1: 静态环境下的车辆调度（无政策限制）")
    print("=" * 60)

    solver = SimpleVRPSolver(customers, depot, vehicle_types)
    routes_q1, vehicle_usage_q1, total_cost_q1 = solver.solve()

    print(f"\n  结果:")
    print(f"  使用车辆数: {len(routes_q1)}")
    print(f"  车辆类型使用: {vehicle_usage_q1}")
    print(f"  总成本: {total_cost_q1:.2f} 元")

    # 可视化
    visualize_solution(customers, depot, routes_q1, 
                    "问题1: 无政策限制配送方案")

    # ============ 新增：成本构成分析 ============
    print("\n  成本构成分析:")

    # 各成本项统计
    STARTUP_COST = 400
    SERVICE_TIME = 20/60
    EARLY_PENALTY = 20   # 元/小时
    LATE_PENALTY = 50    # 元/小时
    CARBON_COST = 0.65   # 元/kg

    total_startup = 0
    total_fuel = 0
    total_electric = 0
    total_carbon_cost = 0
    total_early_penalty = 0
    total_late_penalty = 0
    total_carbon_kg = 0  # 仅碳排放量

    for route in routes_q1:
        if not route.customers:
            continue
        
        total_startup += STARTUP_COST
        
        # 确定车辆类型（根据载重匹配）
        for vt in vehicle_types:
            if vt.max_weight >= route.total_weight:
                vehicle_type = vt
                break
        
        current_time = 8.0
        current_pos = depot
        
        for customer in route.customers:
            dist = current_pos.distance_to(customer)
            speed = get_speed_by_time(current_time)
            
            # 能耗费用
            load_ratio = route.total_weight / vehicle_type.max_weight
            if vehicle_type.energy_type == 'fuel':
                energy_cost, carbon = calculate_fuel_cost(dist, speed, load_ratio)
                total_fuel += energy_cost
            else:
                energy_cost, carbon = calculate_electric_cost(dist, speed, load_ratio)
                total_electric += energy_cost
            
            total_carbon_cost += carbon * CARBON_COST
            total_carbon_kg += carbon
            
            # 更新时间
            travel_t = dist / max(1, speed)
            current_time += travel_t
            
            # 时间窗惩罚
            if current_time < customer.tw_start:
                wait = customer.tw_start - current_time
                total_early_penalty += wait * EARLY_PENALTY
                current_time = customer.tw_start
            elif current_time > customer.tw_end:
                delay = current_time - customer.tw_end
                total_late_penalty += delay * LATE_PENALTY
            
            current_time += SERVICE_TIME
            current_pos = customer
        
        # 返回配送中心
        dist_back = current_pos.distance_to(depot)
        speed = get_speed_by_time(current_time)
        if vehicle_type.energy_type == 'fuel':
            energy_cost, carbon = calculate_fuel_cost(dist_back, speed, 0)
            total_fuel += energy_cost
        else:
            energy_cost, carbon = calculate_electric_cost(dist_back, speed, 0)
            total_electric += energy_cost
        total_carbon_cost += carbon * CARBON_COST

    # 打印
    print(f"  车辆启动成本: {total_startup:.2f} 元")
    print(f"  燃油费用:     {total_fuel:.2f} 元")
    print(f"  电费:         {total_electric:.2f} 元")
    print(f"  碳排放成本:   {total_carbon_cost:.2f} 元")
    print(f"  早到等待成本: {total_early_penalty:.2f} 元")
    print(f"  晚到惩罚成本: {total_late_penalty:.2f} 元")
    print(f"  碳排放量:     {total_carbon_kg:.2f} kg")

    # 绘制成本构成饼图
    fig, ax = plt.subplots(figsize=(8, 6))

    categories = ['启动成本', '燃油费用', '电费', '碳排放成本', '早到等待', '晚到惩罚']
    values = [total_startup, total_fuel, total_electric, total_carbon_cost, 
            total_early_penalty, total_late_penalty]

    # 过滤掉0值的项
    filtered_categories = []
    filtered_values = []
    for cat, val in zip(categories, values):
        if val > 0.01:
            filtered_categories.append(cat)
            filtered_values.append(val)

    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']
    filtered_colors = [colors[i] for i, v in enumerate(values) if v > 0.01]

    wedges, texts, autotexts = ax.pie(
        filtered_values, 
        labels=filtered_categories, 
        colors=filtered_colors, 
        autopct='%1.1f%%',
        startangle=90,
        textprops={'fontsize': 11}
    )

    # 加粗百分比文字
    for autotext in autotexts:
        autotext.set_fontweight('bold')

    ax.set_title(f'问题1: 配送成本构成\n总成本: {total_cost_q1:.2f} 元', 
                fontsize=14, fontweight='bold')

    # 添加图例显示具体金额
    legend_labels = [f'{cat}: {val:.0f}元' for cat, val in zip(filtered_categories, filtered_values)]
    ax.legend(wedges, legend_labels, title="成本明细",
            loc="center left", bbox_to_anchor=(1, 0, 0.5, 1),
            fontsize=10)

    plt.tight_layout()
    plt.show()    
    # ============ 问题2: 绿色配送区限制 ============
    print("\n" + "=" * 60)
    print("问题2: 绿色配送区限行政策（8:00-16:00禁止燃油车进入）")
    print("=" * 60)
    
    # 策略：绿色配送区内客户优先使用新能源车
    # 分离客户
    green_vehicles = [vt for vt in vehicle_types if vt.energy_type == 'electric']
    normal_vehicles = [vt for vt in vehicle_types if vt.energy_type == 'fuel']
    
    # 绿色区内用新能源车
    if in_zone:
        green_solver = SimpleVRPSolver(in_zone, depot, green_vehicles)
        green_routes, green_usage, green_cost = green_solver.solve(n_clusters=3)
    else:
        green_routes, green_usage, green_cost = [], {}, 0
    
    # 绿色区外用所有车辆
    if out_zone:
        normal_solver = SimpleVRPSolver(out_zone, depot, vehicle_types)
        normal_routes, normal_usage, normal_cost = normal_solver.solve()
    else:
        normal_routes, normal_usage, normal_cost = [], {}, 0
    
    all_routes_q2 = green_routes + normal_routes
    total_cost_q2 = green_cost + normal_cost
    
    print(f"\n  结果:")
    print(f"  绿色区内使用新能源车: {len(green_routes)} 辆")
    print(f"  绿色区外使用车辆: {len(normal_routes)} 辆")
    print(f"  总成本: {total_cost_q2:.2f} 元")
    
    # 对比分析
    print(f"\n  === 政策影响分析 ===")
    print(f"  问题1总成本: {total_cost_q1:.2f} 元")
    print(f"  问题2总成本: {total_cost_q2:.2f} 元")
    print(f"  成本变化: {(total_cost_q2 - total_cost_q1):.2f} 元")
    print(f"  成本变化率: {((total_cost_q2 - total_cost_q1) / total_cost_q1 * 100):.2f}%")
    
    # 可视化
    visualize_solution(customers, depot, all_routes_q2,
                       "问题2: 绿色配送区限行方案")
    
    # ============ 问题3: 动态调度策略 ============
    print("\n" + "=" * 60)
    print("问题3: 动态事件响应策略")
    print("=" * 60)
    
    print("""
    动态调度策略设计：
    
    1. 新增订单处理：
       - 评估当前所有车辆的剩余容量
       - 在最近车辆路径中寻找最佳插入位置（最小额外距离）
       - 若无法插入，派遣新车
    
    2. 订单取消处理：
       - 从路径中移除该客户
       - 检查移除后路径是否可合并
    
    3. 地址变更处理：
       - 等同于取消旧订单 + 新增订单
    
    4. 时间窗调整处理：
       - 检查当前到达时间是否满足新时间窗
       - 如不满足，重新插入或调整路线
    
    5. 实时更新机制：
       - 每15分钟检查一次是否有新事件
       - 采用滚动时域优化
    """)
    
    print("=" * 60)
    print("求解完成！")
    print("=" * 60)

if __name__ == "__main__":
    main()