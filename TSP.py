import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.special import factorial
import time
import random
from itertools import permutations
import heapq

# Set up the figure with consistent styling
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = [14, 10]
plt.rcParams['font.size'] = 12

def calculate_tsp_entropy(n, method="brute_force"):
    """
    Calculate entropy for TSP with n cities
    
    Parameters:
    - n: number of cities
    - method: "brute_force" or optimized algorithm name
    
    Returns:
    - entropy value
    """
    if method == "brute_force":
        # Brute force entropy is approximately log(n!)
        if n <= 20:
            return np.log(math.factorial(n-1))
        else:
            # Use Stirling's approximation for large n
            return (n-1) * np.log(n-1) - (n-1)
    elif method == "nearest_neighbor":
        return np.log(n * n)  # O(n²) complexity
    elif method == "dynamic_programming":
        return np.log(n * 2**n)  # O(n * 2^n) complexity
    elif method == "branch_and_bound":
        # Varies based on heuristics, but typically better than brute force
        return np.log(n**2 * 2**(n/2))
    elif method == "genetic_algorithm":
        # Depends on parameters, but generally better than brute force
        return np.log(n**2 * np.log(n))
    else:  # Default for general optimization
        return np.log(n**2)

def estimate_cities_for_time(target_time, method="brute_force", base_time=1e-6):
    """
    Estimate how many cities can be solved within a target computation time
    
    Parameters:
    - target_time: target computation time in seconds
    - method: algorithm method
    - base_time: base computation time per operation
    
    Returns:
    - estimated number of cities
    """
    if method == "brute_force":
        # Solve for n where base_time * (n-1)! ≈ target_time
        n = 4  # Start with small number
        while base_time * math.factorial(n-1) < target_time and n < 20:
            n += 1
        return n
    
    elif method == "nearest_neighbor":
        # O(n²) complexity
        n = int(np.sqrt(target_time / base_time))
        return min(n, 1000)  # Cap at a reasonable number
    
    elif method == "dynamic_programming":
        # O(n * 2^n) complexity
        n = 4
        while base_time * n * (2**n) < target_time and n < 30:
            n += 1
        return n
    
    elif method == "branch_and_bound":
        # Approximate as O(n² * 2^(n/2))
        n = 4
        while base_time * (n**2) * (2**(n/2)) < target_time and n < 50:
            n += 1
        return n
    
    elif method == "genetic_algorithm":
        # Approximate as O(n² * log(n))
        n = int((target_time / base_time)**(1/2.1))  # Slightly more than square root
        return min(n, 10000)  # Cap at a reasonable number
    
    else:
        # Default approximation
        n = int(np.sqrt(target_time / base_time))
        return min(n, 1000)

def generate_random_cities(n, max_coord=100):
    """Generate n random cities in 2D space"""
    return [(random.uniform(0, max_coord), random.uniform(0, max_coord)) for _ in range(n)]

def distance(city1, city2):
    """Calculate Euclidean distance between two cities"""
    return np.sqrt((city1[0] - city2[0])**2 + (city1[1] - city2[1])**2)

def total_distance(route, cities):
    """Calculate total distance of a route"""
    return sum(distance(cities[route[i]], cities[route[i+1]]) for i in range(len(route)-1)) + distance(cities[route[-1]], cities[route[0]])

def brute_force_tsp(cities, time_limit=float('inf')):
    """Solve TSP using brute force approach with time limit"""
    n = len(cities)
    if n > 11:  # Practical limit for brute force
        raise ValueError("Too many cities for brute force approach")
    
    start_time = time.time()
    best_distance = float('inf')
    best_route = None
    operations = 0
    paths_explored = 0
    
    # Fix first city (0) and permute the rest
    for perm in permutations(range(1, n)):
        if time.time() - start_time > time_limit:
            break
            
        route = (0,) + perm
        dist = total_distance(route, cities)
        operations += n  # Approximate operations for distance calculation
        paths_explored += 1
        
        if dist < best_distance:
            best_distance = dist
            best_route = route
    
    end_time = time.time()
    computation_time = end_time - start_time
    
    return {
        "route": best_route,
        "distance": best_distance,
        "time": computation_time,
        "operations": operations,
        "paths_explored": paths_explored,
        "entropy": np.log(paths_explored) if paths_explored > 0 else 0,
        "complete": paths_explored == math.factorial(n-1)
    }

def nearest_neighbor_tsp(cities, time_limit=float('inf')):
    """Solve TSP using nearest neighbor heuristic with time limit"""
    n = len(cities)
    start_time = time.time()
    
    # Start from the first city
    current_city = 0
    route = [current_city]
    unvisited = set(range(1, n))
    operations = 0
    
    while unvisited and (time.time() - start_time <= time_limit):
        next_city = min(unvisited, key=lambda city: distance(cities[current_city], cities[city]))
        operations += len(unvisited)  # Count comparisons
        
        route.append(next_city)
        unvisited.remove(next_city)
        current_city = next_city
    
    # Complete the route if we hit the time limit
    if unvisited:
        route.extend(unvisited)
    
    total_dist = total_distance(route, cities)
    end_time = time.time()
    computation_time = end_time - start_time
    
    return {
        "route": route,
        "distance": total_dist,
        "time": computation_time,
        "operations": operations,
        "entropy": np.log(operations) if operations > 0 else 0,
        "complete": len(unvisited) == 0
    }

def two_opt_tsp(cities, time_limit=float('inf')):
    """Solve TSP using 2-opt improvement heuristic with time limit"""
    n = len(cities)
    start_time = time.time()
    
    # Start with a random route
    route = list(range(n))
    random.shuffle(route)
    best_distance = total_distance(route, cities)
    operations = 0
    improvements = 0
    
    improved = True
    while improved and (time.time() - start_time <= time_limit):
        improved = False
        for i in range(1, n-1):
            for j in range(i+1, n):
                operations += 1
                
                if time.time() - start_time > time_limit:
                    break
                    
                # Try swapping the order of cities between i and j
                new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                new_distance = total_distance(new_route, cities)
                
                if new_distance < best_distance:
                    route = new_route
                    best_distance = new_distance
                    improved = True
                    improvements += 1
                    break
            
            if improved or time.time() - start_time > time_limit:
                break
    
    end_time = time.time()
    computation_time = end_time - start_time
    
    return {
        "route": route,
        "distance": best_distance,
        "time": computation_time,
        "operations": operations,
        "improvements": improvements,
        "entropy": np.log(operations) if operations > 0 else 0,
        "complete": not improved  # Complete if no more improvements possible
    }

def simulated_annealing_tsp(cities, time_limit=float('inf')):
    """Solve TSP using simulated annealing with time limit"""
    n = len(cities)
    start_time = time.time()
    
    # Parameters
    initial_temp = 100.0
    final_temp = 0.1
    alpha = 0.995  # Cooling rate
    
    # Start with a random route
    current_route = list(range(n))
    random.shuffle(current_route)
    current_distance = total_distance(current_route, cities)
    best_route = current_route.copy()
    best_distance = current_distance
    
    temp = initial_temp
    operations = 0
    iterations = 0
    accepted_moves = 0
    
    while temp > final_temp and (time.time() - start_time <= time_limit):
        iterations += 1
        
        # Select two random positions
        i, j = sorted(random.sample(range(n), 2))
        
        # Create new solution by swapping two cities
        new_route = current_route.copy()
        new_route[i:j+1] = reversed(current_route[i:j+1])
        
        # Calculate new distance
        new_distance = total_distance(new_route, cities)
        operations += 1
        
        # Decide whether to accept the new solution
        if new_distance < current_distance:
            current_route = new_route
            current_distance = new_distance
            accepted_moves += 1
            
            if new_distance < best_distance:
                best_route = new_route.copy()
                best_distance = new_distance
        else:
            # Accept worse solution with a probability that decreases with temperature
            p = np.exp((current_distance - new_distance) / temp)
            if random.random() < p:
                current_route = new_route
                current_distance = new_distance
                accepted_moves += 1
        
        # Cool down
        temp *= alpha
    
    end_time = time.time()
    computation_time = end_time - start_time
    
    return {
        "route": best_route,
        "distance": best_distance,
        "time": computation_time,
        "operations": operations,
        "iterations": iterations,
        "accepted_moves": accepted_moves,
        "entropy": np.log(operations) if operations > 0 else 0,
        "complete": temp <= final_temp
    }

def analyze_by_computation_time():
    """Analyze TSP algorithms with fixed computation time budgets"""
    # Define computation time budgets to test
    time_budgets = [0.001, 0.01, 0.1, 1.0, 5.0, 10.0]
    
    # Algorithms to test
    algorithms = [
        {"name": "Brute Force", "func": brute_force_tsp, "color": "#ff6b6b"},
        {"name": "Nearest Neighbor", "func": nearest_neighbor_tsp, "color": "#1dd1a1"},
        {"name": "2-Opt", "func": two_opt_tsp, "color": "#5352ed"},
        {"name": "Simulated Annealing", "func": simulated_annealing_tsp, "color": "#feca57"}
    ]
    
    results = []
    
    for time_budget in time_budgets:
        print(f"\nAnalyzing with time budget: {time_budget} seconds")
        budget_results = {"time_budget": time_budget, "algorithms": []}
        
        for algorithm in algorithms:
            algorithm_name = algorithm["name"]
            print(f"  Testing {algorithm_name}...")
            
            # Estimate appropriate problem size for this time budget
            if algorithm_name == "Brute Force":
                estimated_n = estimate_cities_for_time(time_budget, "brute_force")
            elif algorithm_name == "Nearest Neighbor":
                estimated_n = estimate_cities_for_time(time_budget, "nearest_neighbor")
            elif algorithm_name == "2-Opt":
                estimated_n = min(100, estimate_cities_for_time(time_budget, "nearest_neighbor"))
            elif algorithm_name == "Simulated Annealing":
                estimated_n = min(1000, estimate_cities_for_time(time_budget, "genetic_algorithm"))
            else:
                estimated_n = 20
            
            try:
                cities = generate_random_cities(estimated_n)
                result = algorithm["func"](cities, time_limit=time_budget)
                
                # Calculate theoretical entropy
                theoretical_entropy = calculate_tsp_entropy(
                    estimated_n, 
                    "brute_force" if algorithm_name == "Brute Force" else algorithm_name.lower().replace("-", "_").replace(" ", "_")
                )
                
                # Calculate solution quality compared to estimated optimal
                solution_quality = 1.0  # Base value
                if algorithm_name != "Brute Force" and estimated_n <= 9:
                    # For small enough problems, compare with brute force
                    try:
                        optimal_result = brute_force_tsp(cities)
                        if optimal_result["distance"] > 0:
                            solution_quality = optimal_result["distance"] / max(result["distance"], 1e-10)
                    except:
                        pass
                
                budget_results["algorithms"].append({
                    "name": algorithm_name,
                    "cities": estimated_n,
                    "operations": result["operations"],
                    "measured_entropy": result["entropy"],
                    "theoretical_entropy": theoretical_entropy,
                    "time_used": result["time"],
                    "complete": result["complete"],
                    "solution_quality": solution_quality
                })
                
                print(f"    Completed with {estimated_n} cities, {result['operations']} operations")
                
            except Exception as e:
                print(f"    Error: {str(e)}")
                budget_results["algorithms"].append({
                    "name": algorithm_name,
                    "error": str(e)
                })
        
        results.append(budget_results)
    
    return results

def plot_entropy_by_time(results):
    """Plot entropy achieved within different time budgets"""
    time_budgets = [r["time_budget"] for r in results]
    
    plt.figure(figsize=(14, 8))
    
    # Set up the algorithm properties
    algorithms = [
        {"name": "Brute Force", "marker": "o", "color": "#ff6b6b"},
        {"name": "Nearest Neighbor", "marker": "s", "color": "#1dd1a1"},
        {"name": "2-Opt", "marker": "^", "color": "#5352ed"},
        {"name": "Simulated Annealing", "marker": "D", "color": "#feca57"}
    ]
    
    # Plot measured entropy
    plt.subplot(1, 2, 1)
    for alg in algorithms:
        entropies = []
        for r in results:
            alg_data = next((a for a in r["algorithms"] if a["name"] == alg["name"]), None)
            if alg_data and "measured_entropy" in alg_data:
                entropies.append(alg_data["measured_entropy"])
            else:
                entropies.append(None)
        
        valid_points = [(x, y) for x, y in zip(time_budgets, entropies) if y is not None]
        if valid_points:
            x_vals, y_vals = zip(*valid_points)
            plt.plot(x_vals, y_vals, marker=alg["marker"], color=alg["color"], label=alg["name"])
    
    plt.xscale('log')
    plt.title('Entropy Achieved Within Time Budget')
    plt.xlabel('Computation Time (seconds, log scale)')
    plt.ylabel('Measured Entropy (log scale)')
    plt.grid(True)
    plt.legend()
    
    # Plot cities solved
    plt.subplot(1, 2, 2)
    for alg in algorithms:
        cities = []
        for r in results:
            alg_data = next((a for a in r["algorithms"] if a["name"] == alg["name"]), None)
            if alg_data and "cities" in alg_data:
                cities.append(alg_data["cities"])
            else:
                cities.append(None)
        
        valid_points = [(x, y) for x, y in zip(time_budgets, cities) if y is not None]
        if valid_points:
            x_vals, y_vals = zip(*valid_points)
            plt.plot(x_vals, y_vals, marker=alg["marker"], color=alg["color"], label=alg["name"])
    
    plt.xscale('log')
    plt.yscale('log')
    plt.title('Cities Solvable Within Time Budget')
    plt.xlabel('Computation Time (seconds, log scale)')
    plt.ylabel('Number of Cities (log scale)')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('tsp_entropy_by_time.png', dpi=300)
    plt.show()

def plot_operations_vs_cities(results):
    """Plot operations vs cities for different algorithms"""
    plt.figure(figsize=(10, 6))
    
    # Set up the algorithm properties
    algorithms = [
        {"name": "Brute Force", "marker": "o", "color": "#ff6b6b"},
        {"name": "Nearest Neighbor", "marker": "s", "color": "#1dd1a1"},
        {"name": "2-Opt", "marker": "^", "color": "#5352ed"},
        {"name": "Simulated Annealing", "marker": "D", "color": "#feca57"}
    ]
    
    for alg in algorithms:
        cities = []
        operations = []
        
        for r in results:
            for a in r["algorithms"]:
                if a["name"] == alg["name"] and "cities" in a and "operations" in a:
                    cities.append(a["cities"])
                    operations.append(a["operations"])
        
        if cities and operations:
            plt.scatter(cities, operations, marker=alg["marker"], color=alg["color"], label=alg["name"], alpha=0.7)
    
    plt.yscale('log')
    plt.title('Operations vs Cities for Different Algorithms')
    plt.xlabel('Number of Cities')
    plt.ylabel('Operations (log scale)')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('tsp_operations_vs_cities.png', dpi=300)
    plt.show()

def plot_entropy_cartesian(results):
    """Visualize entropy on a 2D Cartesian space with different computation times"""
    # Create a mesh grid
    x = np.linspace(0, 100, 40)
    y = np.linspace(0, 100, 40)
    X, Y = np.meshgrid(x, y)
    
    # Get data points from results
    time_points = []
    entropy_levels = []
    
    for r in results:
        time_budget = r["time_budget"]
        for alg in r["algorithms"]:
            if "measured_entropy" in alg:
                time_points.append((time_budget, alg["measured_entropy"], alg["name"]))
                entropy_levels.append(alg["measured_entropy"])
    
    # Calculate entropy field - higher entropy in center, decaying outward
    center_x, center_y = 50, 50
    distances = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
    
    # Create an entropy field that looks like the TSP solution space
    entropy_field = 20 - 0.3 * distances
    
    # Plot everything
    plt.figure(figsize=(12, 10))
    
    # Create contour plot of the entropy field
    contour = plt.contourf(X, Y, entropy_field, 20, cmap='plasma', alpha=0.7)
    contour_lines = plt.contour(X, Y, entropy_field, 10, colors='white', alpha=0.5, linewidths=0.5)
    cbar = plt.colorbar(contour, label='Entropy Value')
    
    # Plot the entropy time curves - curves where specific entropy values are achieved at specific times
    # Visualize as contour lines for specific entropy values
    for e_level in sorted(set([round(e, 1) for e in entropy_levels]))[::-1][:5]:  # Take top 5 entropy levels
        if e_level > 0:
            # Find where the entropy field equals this level (approximately)
            level_contour = plt.contour(X, Y, entropy_field, levels=[e_level], 
                                      colors=['#333333'], linewidths=2, linestyles='dashed')
            plt.clabel(level_contour, inline=True, fontsize=9, fmt='S=%.1f' % e_level)
    
    # Add points for each algorithm at their achieved entropy
    markers = {'Brute Force': 'o', 'Nearest Neighbor': 's', '2-Opt': '^', 'Simulated Annealing': 'D'}
    colors = {'Brute Force': '#ff6b6b', 'Nearest Neighbor': '#1dd1a1', '2-Opt': '#5352ed', 'Simulated Annealing': '#feca57'}
    
    # Plot algorithm points arranged around the entropy contours
    for time, entropy, alg_name in time_points:
        if entropy > 0:
            # Find position on the entropy contour
            theta = 0.5 + (np.log10(time) + 3) / 6.0  # Map time to angle (0.5 to 1.5 radians)
            r = 50 - entropy * 2  # Map entropy to radius
            x = center_x + r * np.cos(theta * np.pi)
            y = center_y + r * np.sin(theta * np.pi)
            
            plt.plot(x, y, marker=markers.get(alg_name, 'o'), color=colors.get(alg_name, 'black'), 
                    markersize=10, label=
