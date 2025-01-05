# Conway's Game of Life

##### Table of Contents
- [To Be or Not To Be](#to-be-or-not-to-be)  
- [Getting Started](#getting-started)
- [Code Snippets](#code-snippets)
- [Examples](#examples)


## To Be or Not To Be

Welcome to Conway's beautifully simple and marvelously complex Game of Life!

As we said in the [introduction to this project](https://github.com/dHuberYoumans/maths/blob/main/README.md), the rules are simple:

1. Our world is a flat torus, discretised in quadrilaterals


2. Each square is a cell which can be either _live_ or _dead_. A configuration of live and dead cells is called a generation.


3. Going from one generation to the next, cells die, survive or are (re)born based on the state of their 8 neighbours:

    - Any live cell with fewer than two live neighbours dies, as if by underpopulation
  
    - Any live cell with two or three live neighbours lives on to the next generation
    
    - Any live cell with more than three live neighbours dies, as if by overpopulation
    
    - Any dead cell with exactly three live neighbours becomes a live cell, as if by reproduction

And that's it! 

## Getting Started 

The project contains 3 scripts and a Jupyter notebook, the latter is just a scrapbook to try out ideas. 

#### ```game_of_lfe.py```

This file contains the basic logic of the game. 
It contains the simple cell class `Cell` which contains the basic attribute `Cell.alive` which is a boolean value representing whether the cell is _alive_ (```True```) or _dead_ (```False```).
The main class is `Lattice` which represents the world.
A configuration can be set by the `Lattice.set_states()` method which takes a list of _integer coordinates_ `[...,(i,j),...]` and sets the corresponding cells to _(a)live_.
The `Lattice` class has also a class method `Lattice.plot()` with which the current configuration can be displayed.
The most important method of this class is the update method `Lattice.update()` which passes to the next generation according to the classical rules, which we recalled in the beginning.


#### `gui.py`

This file contains the visual logic, the GUI built with Python's `tkinter` module. 
It creates a window displaying the grid which runs the simulation. 
Each square in the grid represents a cell and their color represents their current status: _white_ for _dead_ and _black_ for _(a)live_.

One can click on the grid to activate cells. 
Once an initial configuration has been chosen, the _Play_ button starts the simmulation. 
The grid is updated according to `Lattice.update()`.

The _Pause_ button allows to pause the simmulation. One can then (de)activate cells and continue the simulation form here by clicking once again the _Play_ button. 

Finally, the _Reset_ button resets the simmulation. 


#### `main.py`

This is the main file. 
Running 

`python main.py`

starts the game!

## Code Snippets

The `Lattice.update()` method:

```python 
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
        EXPLICIT VERSION: [...]
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
```

The `GUI.next_gen()` method:

```python
def next_gen(self):
        self.lattice.update()
        live = self.lattice.get_alive()
        dead = self.lattice.get_dead()

        # update colors
        for (a,b) in live:
            rectangle_id = self.grid[(b,a)] # access via (col,row)
            self.canvas.itemconfig(rectangle_id, fill='black')

        for (a,b) in dead:
            rectangle_id = self.grid[(b,a)] # access via (col,row)
            self.canvas.itemconfig(rectangle_id, fill='white')

        # display count of gens
        self.gens += 1
        self.gen_label.config(text=f"generation: {self.gens}")

        if self.loop:
            self.canvas.after(200, self.next_gen)
        else: # if paused (self.loop = False), update live cells
            self.live_cells = self.get_live_from_lattice() 
```

## Examples

Here are some nice examples from various [starting configurations](https://en.wikipedia.org/wiki/Conway%27s_Game_of_Life)!

**The Toad**
<p align="center">
  <img src="https://github.com/dHuberYoumans/maths/blob/main/GameOfLife/img_README/GoL_Toad.gif" alt="animated" width=500px height=auto />
</p>

**The Glider**
<p align="center">
  <img src="https://github.com/dHuberYoumans/maths/blob/main/GameOfLife/img_README/GoL_Glider.gif" alt="animated" width=500px height=auto />
</p>

**Heavy Weight Space Ship (HWSS)**
<p align="center">
  <img src="https://github.com/dHuberYoumans/maths/blob/main/GameOfLife/img_README/GoL_HWSS.gif" alt="animated" width=500px height=auto />
</p>

And that's it! Happy Hunting!


