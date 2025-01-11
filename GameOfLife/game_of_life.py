import numpy as np
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns
from linalg import *


@dataclass
class Cell():
    alive: bool = False

    def set(self,state):
        self.alive = state

    def get(self):
        return self.alive
    
class SqLattice(): 
    def __init__(self, size: tuple[int,int]):
        self.ROWS, self.COLS = size
        self.grid = self.create_grid()

    def create_grid(self):
        return np.array([[Cell() for _ in range(self.COLS)] for _ in range(self.ROWS)])

    def set_states(self, idx: list[tuple[int,int]], state: bool):
        row_idx = [row for row, _ in idx]
        col_idx = [col for _, col in idx]

        for cell in self.grid[row_idx,col_idx]:
            cell.set(state)

    def get_states(self, grid = None):
        if grid is not None:
            return np.array([[cell.get() for cell in row] for row in grid])
        else:
            return np.array([[cell.get() for cell in row] for row in self.grid])
        
    def get_alive(self):
        return list(zip(*np.where(self.get_states() == True)))
    
    def get_dead(self):
        return list(zip(*np.where(self.get_states() == False)))
    
    def plot(self, ax = None):
        data_ = self.get_states()
        if ax is not None:
            sns.heatmap(data_,cbar=False,linewidths=0.1,xticklabels=False,yticklabels=False,cmap='rocket_r',ax = ax)
        else:
            sns.heatmap(data_,cbar=False,linewidths=0.1,xticklabels=False,yticklabels=False,cmap='rocket_r') 
        
    def clear(self):
        self.grid = self.create_grid()

    def update(self):
        bottom = self.get_states(np.roll(self.grid,-1,axis=0)).astype(int)
        up = self.get_states(np.roll(self.grid,1,axis=0)).astype(int)
        right = self.get_states(np.roll(self.grid,-1,axis=1)).astype(int) 
        left = self.get_states(np.roll(self.grid,1,axis=1)).astype(int)
        upper_right = self.get_states(np.roll(np.roll(self.grid,-1,axis=1),1,axis=0)).astype(int)
        upper_left = self.get_states(np.roll(np.roll(self.grid,1,axis=1),1,axis=0)).astype(int)
        bottom_right = self.get_states(np.roll(np.roll(self.grid,-1,axis=1),-1,axis=0)).astype(int)
        bottom_left = self.get_states(np.roll(np.roll(self.grid,1,axis=1),-1,axis=0)).astype(int)

        live_or_die = bottom + up + right + left + upper_left + upper_right + bottom_left + bottom_right 
      
        # conditions for live cells
        underpopulation = np.where( (live_or_die < 2) & (self.get_states() == True) ) # dies
        overpopulation = np.where( (live_or_die > 3) & (self.get_states() == True) ) # dies
        stable = np.where( ( (live_or_die == 2) | (live_or_die == 3) ) & (self.get_states() == True) ) # lives

        # conditions for dead cells
        reborn = np.where((live_or_die == 3) & (self.get_states() == False)) # lives

        # update
        for cell in self.grid[*underpopulation]:
            cell.set(False)

        for cell in self.grid[*overpopulation]:
            cell.set(False)

        for cell in self.grid[*stable]:
            cell.set(True)

        for cell in self.grid[*reborn]:
            cell.set(True)

        
class HexLattice():
    def __init__(self,rows,cols):
        self.ROWS = rows
        self.COLS = cols
        self.grid = self.create_grid()
        pass

    def create_grid(self):
        # HECS
        self.even_rows = [ [(0,r,c) for c  in range(self.COLS)] for r in range(self.ROWS)]
        self.odd_rows = [ [(1,r,c) for c  in range(self.COLS)] for r in range(self.ROWS)]
        grid = {(a,b,c):Cell() for row in self.even_rows + self.odd_rows for (a,b,c) in row}

        return grid
    
    def set_states(self, idx: list[tuple[int,int,int]], state: bool):
        for (a,b,c) in idx:
            self.grid[(a,b,c)].set(state)

    def get_alive(self):
        return [idx for idx in self.grid.keys() if self.grid[idx].get()]
    
    def get_dead(self):
        return [idx for idx in self.grid.keys() if not self.grid[idx].get()]
    
    def get_states(self):
        return np.array([[[self.grid[(a,r,c)].get() for c in range(self.COLS)] for r in range(self.ROWS)]for a in range(2)])
    
    def clear(self):
        self.grid = self.create_grid()

    def get_neighbours(self,hexagon: tuple[int,int,int]):
        a,r,c = hexagon

        right = (a, r, (c + 1 ) % self.COLS)
        right_above = (1 - a, (r - (1 - a)) % self.ROWS, (c + a) % self.COLS)
        left_above = (1 - a, (r - (1 - a)) % self.ROWS, (c - (1 - a)) % self.COLS)
        left = (a, r, (c - 1) % self.COLS)
        left_below = (1 - a, (r + a) % self.ROWS, (c - (1 - a)) % self.COLS)
        right_below = (1 - a, (r + a) % self.ROWS, (c + a) % self.COLS)

        return (right, right_above, left_above, left, left_below, right_below)

    def update(self):
        current_states = self.get_states()

        for id in self.grid.keys():
            current_cell_state = self.grid[id]
            nbr_state = 0
            nbrs = self.get_neighbours(id)
            for nbr in nbrs:
                nbr_state += current_states[nbr]

            # live cells
            if current_cell_state.get() and nbr_state == 3: # live
                continue
            if current_cell_state.get() and nbr_state < 3: # die, underpopulation
                current_cell_state.set(False)
            if current_cell_state.get() and nbr_state > 3: # die, overpopulation
                current_cell_state.set(False)

            # dead cells
            if not current_cell_state.get() and nbr_state == 3: # live
                current_cell_state.set(True)


# class Lattice():
#     def __init__(self,size: int, type:str):
#         self.size = size
#         self.grid = np.array([[Cell() for i in range(self.size)] for j in range(self.size)])

#     def set_state(self, idx: list[tuple[int,int]], state: bool):
#         row_idx = [row for row, _ in idx]
#         col_idx = [col for _, col in idx]

#         for cell in self.grid[row_idx,col_idx]:
#             cell.set(state)

#     def get_states(self, grid = None):
#         if grid is not None:
#             return np.array([[cell.get() for cell in row] for row in grid])
#         else:
#             return np.array([[cell.get() for cell in row] for row in self.grid])
        
#     def get_alive(self):
#         return list(zip(*np.where(self.get_states() == True)))
    
#     def get_dead(self):
#         return list(zip(*np.where(self.get_states() == False)))
    
#     def plot(self, ax = None):
#         data_ = self.get_states()
#         if ax is not None:
#             sns.heatmap(data_,cbar=False,linewidths=0.1,xticklabels=False,yticklabels=False,cmap='rocket_r',ax = ax)# self.img = ax.imshow(data_, cmap='gray_r', interpolation='nearest') #
#         else:
#             sns.heatmap(data_,cbar=False,linewidths=0.1,xticklabels=False,yticklabels=False,cmap='rocket_r') #self.img = plt.imshow(data_, cmap='gray_r', interpolation='nearest') #sns.heatmap(data_,cbar=False,linewidths=0.1,xticklabels=False,yticklabels=False,cmap='rocket_r')
        

#     def clear(self):
#         self.grid = np.array([[Cell() for i in range(self.size)] for j in range(self.size)])

#     def update(self):
#         live_or_die = np.zeros_like(self.grid)
#         for dir in [-1,1]:
#             for ax in [0,1]:
#                 # direct nbrs: bottom, right, up, left
#                 live_or_die +=  self.get_states(np.roll(self.grid,dir,axis=ax)).astype(int)
#             for ddir in [-1,1]:
#                 # diagonal nbrs: bottom right, upper right, bottom left, upper left
#                 live_or_die += self.get_states(np.roll(np.roll(self.grid,dir,axis=1),ddir,axis=0)).astype(int) 

#         ''' 
#         EXPLICIT VERSION:

#         # # bottom
#         # bottom = self.get_states(np.roll(self.grid,-1,axis=0)).astype(int)
#         # # up
#         # up = self.get_states(np.roll(self.grid,1,axis=0)).astype(int)
#         # # right
#         # right = self.get_states(np.roll(self.grid,-1,axis=1)).astype(int)
#         # # left 
#         # left = self.get_states(np.roll(self.grid,1,axis=1)).astype(int)
#         # # upper right
#         # upper_right = self.get_states(np.roll(np.roll(self.grid,-1,axis=1),1,axis=0)).astype(int)
#         # # upper left
#         # upper_left = self.get_states(np.roll(np.roll(self.grid,1,axis=1),1,axis=0)).astype(int)
#         # # bottom right
#         # bottom_right = self.get_states(np.roll(np.roll(self.grid,-1,axis=1),-1,axis=0)).astype(int)
#         # # bottom left
#         # bottom_left = self.get_states(np.roll(np.roll(self.grid,1,axis=1),-1,axis=0)).astype(int)

#         # live_or_die = bottom + up + right + left + upper_left + upper_right + bottom_left + bottom_right # number of live nbrs
#         '''


#         # update according to Conway's Game of Life
#         # conditions for live cells
#         underpopulation = np.where( (live_or_die < 2) & (self.get_states() == True) ) # dies
#         overpopulation = np.where( (live_or_die > 3) & (self.get_states() == True) ) # dies
#         stable = np.where( ( (live_or_die == 2) | (live_or_die == 3) ) & (self.get_states() == True) ) # lives

#         # conditions for dead cells
#         reborn = np.where((live_or_die == 3) & (self.get_states() == False)) # lives

#         # update
#         for cell in self.grid[*underpopulation]:
#             cell.set(False)

#         for cell in self.grid[*overpopulation]:
#             cell.set(False)

#         for cell in self.grid[*stable]:
#             cell.set(True)

#         for cell in self.grid[*reborn]:
#             cell.set(True)

        
