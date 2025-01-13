import numpy as np
from dataclasses import dataclass
import seaborn as sns
from linalg import *
from scipy.signal import convolve2d
import yaml


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

        # conv kernel for update
        self.nbrhd_kernel = np.array([[1.,1.,1.],[1.,0.,1.],[1.,1.,1.]])

        # rules
        with open('config.yml', 'r') as file:
            config = yaml.safe_load(file)

        self.rule = config["sq_rules"]

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
        current = self.get_states().astype(int)
        live_nbrs = convolve2d(current,self.nbrhd_kernel,mode="same",boundary="wrap")

        survival = np.isin(live_nbrs,self.rule["survival"]) & self.get_states()
        birth = np.isin(live_nbrs, self.rule["birth"]) & ~self.get_states()
        
        stable = np.where( survival ) # life
        not_stable = np.where( ~survival ) # death
        born = np.where(birth) # birth

        # update
        for cell in self.grid[not_stable]:
            cell.set(False)

        for cell in self.grid[stable]:
            cell.set(True)

        for cell in self.grid[born]:
            cell.set(True)

    def rnd_population(self):
        p = np.random.rand(*self.grid.shape) < 0.33
        for r in range(self.ROWS):
                for c in range(self.COLS):
                    self.grid[r, c].set(p[r, c])

class HexLattice():
    def __init__(self,rows,cols):
        self.ROWS = rows
        self.COLS = cols
        self.grid = self.create_grid()

        # for efficient update 
        self.shifts = [
            [(0, 0, -1), (-1, 1, 0), (-1, 1, 1), (0, 0, 1), (-1, 0, 1), (-1, 0, 0)],    # even rows 
            [(0, 0, -1), (-1, 0, -1), (-1, 0,0), (0, 0, 1), (-1, -1, 0), (-1, -1, -1)]  # odd rows
        ]

        # rules
        with open('config.yml', 'r') as file:
            config = yaml.safe_load(file)

        self.rule = config["hex_rules"]


    def create_grid(self):
        # HECS
        grid = np.array( [[[Cell() for c  in range(self.COLS)] for r in range(self.ROWS)] for a in range(2)] )

        return grid
    
    def set_states(self, idx: list[tuple[int,int,int]], state: bool):
        for a,b,c in idx:
            self.grid[a,b,c].set(state)

    def get_alive(self):
        current = self.get_states()
        live = list(zip(*np.where(current)))
        return live #[idx for idx in self.grid.keys() if self.grid[idx].get()]
    
    def get_dead(self):
        current = self.get_states()
        dead = list(zip(*np.where(~current)))
        return dead #[idx for idx in self.grid.keys() if not self.grid[idx].get()]
    
    def get_states(self):
        return np.array([[[self.grid[a,r,c].get() for c in range(self.COLS)] for r in range(self.ROWS)]for a in range(2)])
    
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
        current = self.get_states().astype(int)

        live_nbrs = np.zeros_like(current)
        for a in range(2): 
            for shift in self.shifts[a]:
                live_nbrs[a] += np.roll(current, shift=shift, axis=(0,1,2))[a]

        survival = np.isin(live_nbrs,self.rule["survival"]) & self.get_states()
        birth = np.isin(live_nbrs, self.rule["birth"]) & ~self.get_states()
        
        stable = np.where( survival ) # life
        not_stable = np.where( ~survival ) # death
        born = np.where(birth) # birth

        # update
        for cell in self.grid[not_stable]:
            cell.set(False)
        
        for cell in self.grid[stable]:
            cell.set(True)

        for cell in self.grid[born]:
            cell.set(True)

    # def update(self):
    #     current_states = self.get_states()
    #     tmp = self.grid.copy()

    #     for id in self.grid.keys():
    #         current_cell_state = self.grid[id]
    #         nbr_state = 0
    #         nbrs = self.get_neighbours(id)

    #         for nbr in nbrs:
    #             nbr_state += current_states[*nbr]

    #         if current_cell_state.get() and (nbr_state == 3 or nbr_state == 4): # survive
    #             continue
    #         elif not current_cell_state.get() and nbr_state == 2: # reborn
    #             tmp[id].set(True)
    #         else:
    #             tmp[id].set(False) # die

    #     self.grid = tmp

    def rnd_population(self):
        # for cell_id in self.grid:
        #     p = np.random.uniform(0,1)
        #     if p < 0.33:
        #         self.grid[cell_id].set(True)
        p = np.random.rand(2, self.ROWS, self.COLS) < 0.33
        for a in range(2):
            for r in range(self.ROWS):
                for c in range(self.COLS):
                    self.grid[a, r, c].set(p[a, r, c])
        
