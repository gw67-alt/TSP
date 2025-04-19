import numpy as np
import matplotlib.pyplot as plt
import time
import random
import math
from itertools import permutations

# TSP Core Functions
def generate_random_cities(n, max_coord=100):
    """Generate n random cities in 2D space"""
    return [(random.uniform(0, max_coord), random.uniform(0, max_coord)) for _ in range(n)]

def calculate_distance(city1, city2):
    """Calculate Euclidean distance between two cities"""
    return math.sqrt((city1[0] - city2[0])**2 + (city1[1] - city2[1])**2)

def total_distance(route, cities):
    """Calculate total distance of a route, including return to start"""
    return sum(calculate_distance(cities[route[i]], cities[route[i+1]]) for i in range(len(route)-1)) + calculate_distance(cities[route[-1]], cities[route[0]])

# TSP Algorithms
def nearest_neighbor_tsp(cities, time_limit=float('inf')):
    """Solve TSP using nearest neighbor heuristic"""
    n = len(cities)
    start_time = time.time()
    
    current_city = 0
    route = [current_city]
    unvisited = set(range(1, n))
    
    while unvisited and (time.time() - start_time <= time_limit):
        next_city = min(unvisited, key=lambda city: calculate_distance(cities[current_city], cities[city]))
        route.append(next_city)
        unvisited.remove(next_city)
        current_city = next_city
    
    # Complete the route if we hit the time limit
    if unvisited:
        route.extend(unvisited)
    
    return {
        "route": route,
        "distance": total_distance(route, cities),
        "time": time.time() - start_time,
        "algorithm": "Nearest Neighbor"
    }

def two_opt_tsp(cities, time_limit=float('inf'), start_route=None):
    """Solve TSP using 2-opt improvement heuristic"""
    n = len(cities)
    start_time = time.time()
    
    # Start with provided route or a random route
    if start_route:
        route = start_route.copy()
    else:
        route = list(range(n))
        random.shuffle(route)
        
    best_distance = total_distance(route, cities)
    
    improved = True
    while improved and (time.time() - start_time <= time_limit):
        improved = False
        for i in range(1, n-1):
            for j in range(i+1, n):
                if time.time() - start_time > time_limit:
                    break
                
                # Try 2-opt swap
                new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                new_distance = total_distance(new_route, cities)
                
                if new_distance < best_distance:
                    route = new_route
                    best_distance = new_distance
                    improved = True
                    break
            
            if improved or time.time() - start_time > time_limit:
                break
    
    return {
        "route": route,
        "distance": best_distance,
        "time": time.time() - start_time,
        "algorithm": "Two-Opt"
    }

def nn_plus_two_opt(cities, time_limit=float('inf')):
    """Hybrid approach: Start with Nearest Neighbor then improve with 2-opt"""
    start_time = time.time()
    
    # Allocate 1/3 of time for NN
    nn_time_limit = time_limit / 3 if not math.isinf(time_limit) else float('inf')
    nn_result = nearest_neighbor_tsp(cities, time_limit=nn_time_limit)
    
    # Calculate remaining time for 2-opt
    elapsed = time.time() - start_time
    remaining_time = max(0.001, time_limit - elapsed) if not math.isinf(time_limit) else float('inf')
    
    # Use 2-opt to improve NN result
    two_opt_result = two_opt_tsp(cities, time_limit=remaining_time, start_route=nn_result["route"])
    
    return {
        "route": two_opt_result["route"],
        "distance": two_opt_result["distance"],
        "time": time.time() - start_time,
        "algorithm": "NN + Two-Opt"
    }

def brute_force_tsp(cities, time_limit=float('inf')):
    """Solve TSP using brute force approach"""
    n = len(cities)
    if n > 11:  # Practical limit for brute force
        raise ValueError("Too many cities for brute force approach")
        
    start_time = time.time()
    best_distance = float('inf')
    best_route = None
    
    # Fix first city (0) and permute the rest
    for perm in permutations(range(1, n)):
        if time.time() - start_time > time_limit:
            break
            
        route = [0] + list(perm)
        dist = total_distance(route, cities)
        
        if dist < best_distance:
            best_distance = dist
            best_route = route
    
    return {
        "route": best_route,
        "distance": best_distance,
        "time": time.time() - start_time,
        "algorithm": "Brute Force"
    }

# The Combinatoric Configuration Estimator

def combinatoric_factor(n, time_available):
    """
    Calculate a single combinatoric factor that represents the complexity-vs-time ratio
    
    Parameters:
    - n: Number of cities
    - time_available: Available computation time in seconds
    
    Returns:
    - combinatoric_factor: A single value between 0 and 100 that encapsulates the problem complexity
    """
    # Calculate factorial growth for reference
    if n <= 12:
        factorial_growth = math.factorial(n)
    else:
        # Use Stirling's approximation for large n
        factorial_growth = math.sqrt(2 * math.pi * n) * (n / math.e)**n
    
    # Calculate quadratic growth for reference
    quadratic_growth = n * n
    
    # Calculate theoretical operations possible in the given time
    # Assume a processor can do roughly 10M operations per second
    operations_possible = time_available * 1e7
    
    # Calculate the complexity ratio (how many times we can 'solve' the problem in the given time)
    factorial_ratio = operations_possible / factorial_growth if factorial_growth > 0 else float('inf')
    quadratic_ratio = operations_possible / quadratic_growth if quadratic_growth > 0 else float('inf')
    
    # Blend the ratios based on problem size
    if n <= 10:
        # For small problems, factorial dominates
        complexity_ratio = factorial_ratio
    elif n <= 20:
        # For medium problems, blend factorial and quadratic
        blend_factor = (n - 10) / 10  # 0 at n=10, 1 at n=20
        complexity_ratio = (1 - blend_factor) * factorial_ratio + blend_factor * quadratic_ratio
    else:
        # For large problems, quadratic dominates
        complexity_ratio = quadratic_ratio
    
    # Convert to logarithmic scale and normalize to 0-100 range
    if complexity_ratio <= 0:
        return 0
    elif complexity_ratio >= 1e10:
        return 100
    else:
        # Log scale conversion
        log_ratio = math.log10(complexity_ratio)
        # Map from approximately -10 to 10 to 0 to 100
        normalized_factor = (log_ratio + 10) * 5
        # Clamp to 0-100 range
        return max(0, min(100, normalized_factor))

def select_algorithm_by_factor(combinatoric_factor, n):
    """
    Select the best TSP algorithm based on the combinatoric factor
    
    Parameters:
    - combinatoric_factor: The single value between 0 and 100
    - n: Number of cities (used for special cases)
    
    Returns:
    - algorithm_name: Name of the selected algorithm
    - configuration: Dictionary with additional configuration parameters
    """
    # Special case for very small problems
    if n <= 8 and combinatoric_factor >= 50:
        return "Brute Force", {}
    
    # For most problems, use the combinatoric factor to select an algorithm
    if combinatoric_factor < 20:
        # Very limited computational resources compared to problem size
        return "Nearest Neighbor", {}
    elif combinatoric_factor < 40:
        # Limited resources, but enough for simple improvements
        return "Nearest Neighbor", {"multiple_starts": min(5, n // 2)}
    elif combinatoric_factor < 60:
        # Moderate resources
        return "NN + Two-Opt", {"local_search_iterations": min(1000, n * n)}
    elif combinatoric_factor < 80:
        # Good resources
        return "NN + Two-Opt", {"local_search_iterations": min(5000, n * n * 2)}
    else:
        # Abundant resources
        return "NN + Two-Opt", {"local_search_iterations": min(10000, n * n * 5)}

def solve_tsp_with_estimator(cities, time_available):
    """
    Solve the TSP problem using the combinatoric factor estimator
    
    Parameters:
    - cities: List of (x, y) coordinates
    - time_available: Available computation time in seconds
    
    Returns:
    - result: Dictionary with solution details
    """
    n = len(cities)
    start_time = time.time()
    
    # Calculate the combinatoric factor
    factor = combinatoric_factor(n, time_available)
    
    # Select algorithm based on the factor
    algorithm_name, config = select_algorithm_by_factor(factor, n)
    
    print(f"Problem size: {n} cities")
    print(f"Available time: {time_available:.4f} seconds")
    print(f"Combinatoric factor: {factor:.2f}")
    print(f"Selected algorithm: {algorithm_name}")
    print(f"Configuration: {config}")
    
    # Run the selected algorithm
    if algorithm_name == "Brute Force":
        result = brute_force_tsp(cities, time_limit=time_available)
    
    elif algorithm_name == "Nearest Neighbor":
        if "multiple_starts" in config and config["multiple_starts"] > 1:
            # Run NN from multiple starting points
            best_distance = float('inf')
            best_route = None
            
            # Choose starting cities
            starting_points = random.sample(range(n), min(config["multiple_starts"], n))
            
            for start_city in starting_points:
                # Run NN from this starting point
                current = start_city
                route = [current]
                unvisited = set(range(n))
                unvisited.remove(current)
                
                while unvisited:
                    next_city = min(unvisited, key=lambda city: calculate_distance(cities[current], cities[city]))
                    route.append(next_city)
                    unvisited.remove(next_city)
                    current = next_city
                
                distance = total_distance(route, cities)
                
                if distance < best_distance:
                    best_distance = distance
                    best_route = route
            
            result = {
                "route": best_route,
                "distance": best_distance,
                "time": time.time() - start_time,
                "algorithm": f"Nearest Neighbor (Multiple Starts: {config['multiple_starts']})"
            }
        else:
            # Standard NN
            result = nearest_neighbor_tsp(cities, time_limit=time_available)
    
    elif algorithm_name == "NN + Two-Opt":
        # Get NN solution
        nn_result = nearest_neighbor_tsp(cities, time_limit=time_available / 3)
        
        # Calculate remaining time
        elapsed = time.time() - start_time
        remaining_time = max(0.001, time_available - elapsed)
        
        # Improve with limited 2-opt iterations
        iterations = config.get("local_search_iterations", n * n)
        
        # Start with NN route
        route = nn_result["route"]
        best_distance = total_distance(route, cities)
        
        # 2-opt improvement with controlled iterations
        improved = True
        iteration_count = 0
        
        while improved and iteration_count < iterations and (time.time() - start_time <= time_available):
            improved = False
            
            for i in range(1, n-1):
                if time.time() - start_time > time_available:
                    break
                    
                for j in range(i+1, n):
                    iteration_count += 1
                    
                    if iteration_count > iterations or time.time() - start_time > time_available:
                        break
                    
                    # Try 2-opt swap
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    new_distance = total_distance(new_route, cities)
                    
                    if new_distance < best_distance:
                        route = new_route
                        best_distance = new_distance
                        improved = True
                        break
                
                if improved or time.time() - start_time > time_available:
                    break
        
        result = {
            "route": route,
            "distance": best_distance,
            "time": time.time() - start_time,
            "algorithm": f"NN + Two-Opt (Iterations: {iteration_count})"
        }
    
    else:
        # Default to NN
        result = nearest_neighbor_tsp(cities, time_limit=time_available)
    
    # Add the combinatoric factor to the result
    result["combinatoric_factor"] = factor
    
    return result

def visualize_combinatoric_factor():
    """Visualize how the combinatoric factor changes with problem size and time"""
    problem_sizes = range(5, 101, 5)
    time_budgets = [0.01, 0.1, 1.0, 10.0]
    
    # Calculate factors
    factors = {}
    for t in time_budgets:
        factors[t] = [combinatoric_factor(n, t) for n in problem_sizes]
    
    # Plot
    plt.figure(figsize=(12, 8))
    for t in time_budgets:
        plt.plot(problem_sizes, factors[t], marker='o', linewidth=2, label=f'Time Budget: {t}s')
    
    plt.title('Combinatoric Factor by Problem Size and Time Budget', fontsize=14)
    plt.xlabel('Number of Cities', fontsize=12)
    plt.ylabel('Combinatoric Factor (0-100)', fontsize=12)
    plt.grid(True)
    plt.legend(fontsize=10)
    
    # Add algorithm selection regions
    plt.axhspan(0, 20, alpha=0.2, color='red', label='NN')
    plt.axhspan(20, 40, alpha=0.2, color='orange', label='NN Multiple Starts')
    plt.axhspan(40, 60, alpha=0.2, color='yellow', label='NN + Two-Opt Basic')
    plt.axhspan(60, 80, alpha=0.2, color='green', label='NN + Two-Opt Extended')
    plt.axhspan(80, 100, alpha=0.2, color='blue', label='NN + Two-Opt Intensive')
    
    plt.savefig('combinatoric_factor_chart.png', dpi=300)
    plt.show()

def main():
    """Main function to demonstrate the combinatoric estimator"""
    print("TSP Combinatoric Configuration Estimator")
    print("=" * 60)
    
    # Visualize the combinatoric factor
    print("\nGenerating Combinatoric Factor Visualization...")
    visualize_combinatoric_factor()
    
    # Test with different problem sizes and time budgets
    test_cases = [
        (10, 0.01),  # Small problem, tiny time
        (10, 1.0),   # Small problem, generous time
        (20, 0.1),   # Medium problem, small time
        (20, 2.0),   # Medium problem, generous time
        (50, 0.1),   # Large problem, small time
        (150, 5.0)    # Large problem, generous time
    ]
    
    for n, time_budget in test_cases:
        print("\n" + "=" * 60)
        print(f"Testing with {n} cities and {time_budget}s time budget")
        
        # Generate random cities
        cities = generate_random_cities(n)
        
        # Solve using the estimator
        result = solve_tsp_with_estimator(cities, time_budget)
        
        print(f"Solution found:")
        print(f"  Distance: {result['distance']:.2f}")
        print(f"  Time used: {result['time']:.4f}s")
        print(f"  Algorithm: {result['algorithm']}")
    
    print("\nEstimator test complete!")

if __name__ == "__main__":
    main()
