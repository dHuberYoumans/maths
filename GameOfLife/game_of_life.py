import numpy as np
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns

@dataclass
class Cell():
    alive: bool = False

    def set(self,state):
        self.alive = state

    def get(self):
        return self.alive
    
class Lattice():
    def __init__(self,size: int):
        self.size = size
        self.grid = np.array([[Cell() for i in range(self.size)] for j in range(self.size)])

    def set_state(self, idx: list[tuple[int,int]], state: bool):
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
            sns.heatmap(data_,cbar=False,linewidths=0.1,xticklabels=False,yticklabels=False,cmap='rocket_r',ax = ax)# self.img = ax.imshow(data_, cmap='gray_r', interpolation='nearest') #
        else:
            sns.heatmap(data_,cbar=False,linewidths=0.1,xticklabels=False,yticklabels=False,cmap='rocket_r') #self.img = plt.imshow(data_, cmap='gray_r', interpolation='nearest') #sns.heatmap(data_,cbar=False,linewidths=0.1,xticklabels=False,yticklabels=False,cmap='rocket_r')
        

    def clear(self):
        self.grid = np.array([[Cell() for i in range(self.size)] for j in range(self.size)])

    def update(self):
        live_or_die = np.zeros_like(self.grid)
        for dir in [-1,1]:
            for ax in [0,1]:
                # direct nbrs: bottom, right, up, left
                live_or_die +=  self.get_states(np.roll(self.grid,dir,axis=ax)).astype(int)
            for ddir in [-1,1]:
                # diagonal nbrs: bottom right, upper right, bottom left, upper left
                live_or_die += self.get_states(np.roll(np.roll(self.grid,dir,axis=1),ddir,axis=0)).astype(int) 

        ''' 
        EXPLICIT VERSION:

        # # bottom
        # bottom = self.get_states(np.roll(self.grid,-1,axis=0)).astype(int)
        # # up
        # up = self.get_states(np.roll(self.grid,1,axis=0)).astype(int)
        # # right
        # right = self.get_states(np.roll(self.grid,-1,axis=1)).astype(int)
        # # left 
        # left = self.get_states(np.roll(self.grid,1,axis=1)).astype(int)
        # # upper right
        # upper_right = self.get_states(np.roll(np.roll(self.grid,-1,axis=1),1,axis=0)).astype(int)
        # # upper left
        # upper_left = self.get_states(np.roll(np.roll(self.grid,1,axis=1),1,axis=0)).astype(int)
        # # bottom right
        # bottom_right = self.get_states(np.roll(np.roll(self.grid,-1,axis=1),-1,axis=0)).astype(int)
        # # bottom left
        # bottom_left = self.get_states(np.roll(np.roll(self.grid,1,axis=1),-1,axis=0)).astype(int)

        # live_or_die = bottom + up + right + left + upper_left + upper_right + bottom_left + bottom_right # number of live nbrs
        '''


        # update according to Conway's Game of Life
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

        
