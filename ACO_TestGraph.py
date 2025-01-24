import numpy as np
import random


# گراف نمونه به صورت ماتریس مجاورت با اسم شهرها
cities = ["Tehran", "Mashhad", "Esfahan", "Shiraz", "Kurdistan"]
graph = [
    [0, 2, 2, 5, 0],
    [2, 0, 3, 0, 3],
    [2, 3, 0, 2, 3],
    [5, 0, 2, 0, 2],
    [0, 3, 3, 2, 0]
]

# پارامترهای الگوریتم ACO
alpha = 1  # تاثیر فرمون
beta = 2   # تاثیر فاصله
rho = 0.5  # نرخ تبخیر فرمون
num_ants = 5
num_iterations = 100
num_nodes = len(graph)

# مقداردهی اولیه فرمون‌ها
pheromone = np.ones((num_nodes, num_nodes))

# تابع احتمال حرکت مورچه
def calculate_probabilities(current_node, visited):
    probs = []
    for next_node in range(num_nodes):
        if next_node not in visited and graph[current_node][next_node] > 0:
            tau = pheromone[current_node][next_node] ** alpha
            eta = (1 / graph[current_node][next_node]) ** beta
            probs.append(tau * eta)
        else:
            probs.append(0)
    total = sum(probs)
    return [p / total if total > 0 else 0 for p in probs]

# بروزرسانی فرمون‌ها
def update_pheromones(paths, path_costs):
    global pheromone
    pheromone *= (1 - rho)  # تبخیر فرمون
    for path, cost in zip(paths, path_costs):
        for i in range(len(path) - 1):
            pheromone[path[i]][path[i + 1]] += 1 / cost
            pheromone[path[i + 1]][path[i]] += 1 / cost  # برای گراف غیرجهت‌دار

# الگوریتم اصلی ACO
def ant_colony_optimization():
    best_path = None
    best_cost = float('inf')

    for iteration in range(num_iterations):
        paths = []
        path_costs = []

        # حرکت هر مورچه
        for ant in range(num_ants):
            current_node = random.randint(0, num_nodes - 1)
            visited = [current_node]
            while len(visited) < num_nodes:
                probabilities = calculate_probabilities(current_node, visited)
                if sum(probabilities) == 0:
                    break  # جلوگیری از گیر کردن مورچه
                next_node = random.choices(range(num_nodes), probabilities)[0]
                visited.append(next_node)
                current_node = next_node
            # برگشت به خانه اولیه
            if len(visited) == num_nodes:
                visited.append(visited[0])
                paths.append(visited)

                # محاسبه هزینه مسیر
                cost = sum(graph[visited[i]][visited[i + 1]] for i in range(len(visited) - 1))
                path_costs.append(cost)

                # به روز رسانی بهترین مسیر
                if cost < best_cost:
                    best_cost = cost
                    best_path = visited

        # بروزرسانی فرمون‌ها فقط در صورتی که مسیرهای معتبری وجود داشته باشد
        if paths:
            update_pheromones(paths, path_costs)

        print(f"Iteration {iteration + 1}: Best cost = {best_cost}")

    # تبدیل بهترین مسیر به نام شهرها
    if best_path:
        best_path_cities = [cities[node] for node in best_path]
        return best_path_cities, best_cost
    else:
        return None, None

# اجرای الگوریتم
best_path, best_cost = ant_colony_optimization()
if best_path:
    print(f"Best Path: {' -> '.join(best_path)}")
    print(f"Best Cost: {best_cost}")
else:
    print("No valid path found.")
