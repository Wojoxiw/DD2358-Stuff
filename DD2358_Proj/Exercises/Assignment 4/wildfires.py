import numpy as np
import matplotlib.pyplot as plt
import random
import pyvtk
import dask
from dask import delayed
import dask.array as da

# Constants
GRID_SIZE = 800  # 800x800 forest grid
FIRE_SPREAD_PROB = 0.3  # Probability that fire spreads to a neighboring tree
BURN_TIME = 3  # Time before a tree turns into ash
DAYS = 60  # Maximum simulation time

# State definitions
EMPTY = 0    # No tree
TREE = 1     # Healthy tree 
BURNING = 2  # Burning tree 
ASH = 3      # Burned tree 

def initialize_forest():
    """Creates a forest grid with all trees and ignites one random tree."""
    forest = np.ones((GRID_SIZE, GRID_SIZE), dtype=int)  # All trees
    burn_time = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)  # Tracks how long a tree burns
    
    # Ignite a random tree
    x, y = random.randint(0, GRID_SIZE-1), random.randint(0, GRID_SIZE-1)
    forest[x, y] = BURNING
    burn_time[x, y] = 1  # Fire starts burning
    
    return forest, burn_time

def get_neighbors(x, y):
    """Returns the neighboring coordinates of a cell in the grid."""
    neighbors = []
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:  # Up, Down, Left, Right
        nx, ny = x + dx, y + dy
        if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE:
            neighbors.append((nx, ny))
    return neighbors

def simulate_wildfire_serial(dummyInput = 0, plotting = False):
    """Simulates wildfire spread over time."""
    forest, burn_time = initialize_forest()
    
    fire_spread = []  # Track number of burning trees each day
    
    for day in range(DAYS):
        new_forest = forest.copy()
        
        for x in range(GRID_SIZE):
            for y in range(GRID_SIZE):
                if forest[x, y] == BURNING:
                    burn_time[x, y] += 1  # Increase burn time
                    
                    # If burn time exceeds threshold, turn to ash
                    if burn_time[x, y] >= BURN_TIME:
                        new_forest[x, y] = ASH
                    
                    # Spread fire to neighbors
                    for nx, ny in get_neighbors(x, y):
                        if forest[nx, ny] == TREE and random.random() < FIRE_SPREAD_PROB:
                            new_forest[nx, ny] = BURNING
                            burn_time[nx, ny] = 1
        
        forest = new_forest.copy()
        fire_spread.append(np.sum(forest == BURNING))
        
        if np.sum(forest == BURNING) == 0:  # Stop if no more fire
            break
        
        if plotting:
            # Plot grid every 5 days
            if day % 5 == 0 or day == DAYS - 1:
                plt.figure(figsize=(6, 6))
                plt.imshow(forest, cmap='viridis', origin='upper')
                plt.title(f"Wildfire Spread - Day {day}")
                plt.colorbar(label="State: 0=Empty, 1=Tree, 2=Burning, 3=Ash")
                plt.show()
    
    return fire_spread

#@delayed
def simulate_wildfire_dask(chunkSize = GRID_SIZE, i = 0, VTKing = False):
    """Simulates wildfire spread over time."""
    
    forest = np.ones((GRID_SIZE, GRID_SIZE), dtype=int)#, chunks = (chunkSize, chunkSize))  # All trees
    burn_time = np.ones((GRID_SIZE, GRID_SIZE), dtype=int)#, chunks = (chunkSize, chunkSize))  # Tracks how long a tree burns
    
    # Ignite a random tree
    x, y = random.randint(0, GRID_SIZE-1), random.randint(0, GRID_SIZE-1)
    forest[x, y] = BURNING
    burn_time = np.abs(burn_time - forest)  # Fire starts burning... hopefully this is faster?
        

    fire_spread = np.zeros(DAYS, dtype=int)  # Track number of burning trees each day
    
    if(VTKing):
        filename = f'forest{0:03d}.vtk'
        vtk_data = pyvtk.VtkData(
            pyvtk.StructuredPoints([GRID_SIZE, GRID_SIZE, 1]),  # 2D structured grid
            pyvtk.PointData(pyvtk.Scalars(forest.flatten(), name="forest"),)  # The current condition of the forest
        )
        vtk_data.tofile(filename)
    
    for day in range(DAYS):
        new_forest = forest.copy()
        
        for x in range(GRID_SIZE):
            for y in range(GRID_SIZE):
                if forest[x, y] == BURNING:
                    burn_time[x, y] += 1  # Increase burn time
                    
                    # If burn time exceeds threshold, turn to ash
                    if burn_time[x, y] >= BURN_TIME:
                        new_forest[x, y] = ASH
                    
                    # Spread fire to neighbors
                    for nx, ny in get_neighbors(x, y):
                        if forest[nx, ny] == TREE and random.random() < FIRE_SPREAD_PROB:
                            new_forest[nx, ny] = BURNING
                            burn_time[nx, ny] = 1
        
        forest = new_forest.copy()
        fire_spread[day] = np.sum(forest == BURNING)
        
        if(VTKing):
            filename = f'forest{(day+1):03d}.vtk'
            vtk_data = pyvtk.VtkData(
                pyvtk.StructuredPoints([GRID_SIZE, GRID_SIZE, 1]),  # 2D structured grid
                pyvtk.PointData(pyvtk.Scalars(forest.flatten(), name="forest"),)  # The current condition of the forest
            )
            vtk_data.tofile(filename)
        
        if np.sum(forest == BURNING) == 0:  # Stop if no more fire
            break
        

    
    return da.from_array(fire_spread, chunks = (chunkSize))