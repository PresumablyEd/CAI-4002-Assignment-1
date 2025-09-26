# solver_impl.py
import heapq
from typing import List, Tuple, Set, Dict, Optional

# Define the Grid type
Grid = Tuple[int, ...]

def manhattan_distance(state: Grid) -> int:
    """Calculate Manhattan distance heuristic."""
    total_distance = 0
    for i, tile in enumerate(state):
        if tile != 0:
            target_row = (tile - 1) // 3
            target_col = (tile - 1) % 3
            current_row = i // 3
            current_col = i % 3
            distance = abs(target_row - current_row) + abs(target_col - current_col)
            total_distance += distance
    return total_distance

def get_neighbors(state: Grid) -> List[Grid]:
    """Get neighbors of the current state."""
    neighbors = []
    z = state.index(0)
    # Adjacency mapping for 8-puzzle
    adj = {
        0: [1, 3],
        1: [0, 2, 4],
        2: [1, 5],
        3: [0, 4, 6],
        4: [1, 3, 5, 7],
        5: [2, 4, 8],
        6: [3, 7],
        7: [4, 6, 8],
        8: [5, 7]
    }
    for i in adj[z]:
        new_state = list(state)
        new_state[z], new_state[i] = new_state[i], new_state[z]
        neighbors.append(tuple(new_state))
    return neighbors

def reconstruct_path(came_from: Dict[Grid, Grid], current: Grid) -> List[Grid]:
    """Reconstruct path from start to end."""
    path = []
    while current in came_from:
        path.append(current)
        current = came_from[current]
    path.append(current)
    return path[::-1]

def solve_puzzle(start: Grid) -> List[Grid]:
    """Solve the puzzle using A* algorithm with Manhattan distance."""
    # Set to track visited states
    visited: Set[Grid] = set()
    # Set to track states in the frontier
    frontier_set: Set[Grid] = set()
    # Dictionary to track path
    came_from: Dict[Grid, Grid] = {}
    # g_score for current state (cost)
    g_score: Dict[Grid, int] = {start: 0}
    # Priority queue (frontier) with (f_score, state)
    priority_queue: List[Tuple[int, Grid]] = [(0, start)]
    frontier_set.add(start)
    
    goal = (1, 2, 3, 4, 5, 6, 7, 8, 0)
    
    while priority_queue:
        # Pop the state with the lowest f_score
        _, current = heapq.heappop(priority_queue)
        frontier_set.remove(current)
        
        # If we've reached the goal, return the path
        if current == goal:
            return reconstruct_path(came_from, current)
        
        # Mark the current state as visited
        visited.add(current)
        
        # Explore neighbors
        for neighbor in get_neighbors(current):
            if neighbor in visited:
                continue
            
            # Calculate g_score for neighbor
            new_g_score = g_score[current] + 1
            
            # If we already have a better path, skip
            if neighbor in g_score and new_g_score >= g_score[neighbor]:
                continue
            
            # Update the path
            came_from[neighbor] = current
            g_score[neighbor] = new_g_score
            # Calculate f_score
            f_score = new_g_score + manhattan_distance(neighbor)
            
            # Add to frontier if it hasn't been visited
            if neighbor not in frontier_set:
                heapq.heappush(priority_queue, (f_score, neighbor))
                frontier_set.add(neighbor)
    
    # If we've exhausted the search space, return empty path
    return []