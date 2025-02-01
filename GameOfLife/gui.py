from game_of_life import *
from linalg import *
import tkinter as tk
import numpy as np
import yaml



class GUI():
    def __init__(self, rows: int, cols: int, off: int = 120):
        with open('config.yml', 'r') as file:
            config = yaml.safe_load(file)

        # layout 
        self.X =config["layout"]["geometry"]["WIDTH"]
        self.Y = config["layout"]["geometry"]["HEIGHT"]
        self.off = off
        self.root = tk.Tk()
        self.root.title("Conway's Game of Life")
        self.root.geometry(f'{self.X+self.off}x{self.Y+self.off}')

        # colors
        self.C_LIVE = config["layout"]["colors"]["LIVE"]
        self.C_DEAD = config["layout"]["colors"]["DEAD"]
        self.C_CELL_OUTLINE = config["layout"]["colors"]["CELL_OUTLINE"]
        self.C_CANVAS_BG = config["layout"]["colors"]["CANVAS_BG"]

        # update frequency
        self.update_freq = 100 # ms
        
        # lattice type
        self.lattice_type = None

        # grid size
        self.ROWS = rows
        self.COLS = cols

        # cells
        self.cells_by_id = {} # cell ids <-> coord
        self.live_cells = []
        
        # gens
        self.gens = 0

        # loop
        self.loop = True

        self.create_layout()

        self.cells_by_pos = None

    def create_canvas(self):
        self.canvas = tk.Canvas(self.root, width=self.X, height=self.Y, bg=self.C_CANVAS_BG)
        self.canvas.pack(padx=10,pady=10)

    def create_layout(self):
        self.create_canvas()

        # display current generation
        self.gen_label = tk.Label(self.root,text=f"generations: {self.gens}")
        self.gen_label.pack()

        # buttons
        button_frame = tk.Frame(self.root)
        button_frame.pack(pady=10)  # Add some vertical padding

        # play button
        self.button_play = tk.Button(button_frame,text="Play",command=self.play)
        self.button_play.pack(side=tk.LEFT,padx=5)

        # stop button
        self.button_stop = tk.Button(button_frame,text="Pause",command=self.pause)
        self.button_stop.pack(side=tk.LEFT,padx=5)

        # next generation
        self.button_reset = tk.Button(button_frame,text="Next",command=self.next)
        self.button_reset.pack(side=tk.LEFT,padx=5)

        # random initial configuration
        self.button_reset = tk.Button(button_frame,text="Random",command=self.random)
        self.button_reset.pack(side=tk.LEFT,padx=5)

        # reset button
        self.button_reset = tk.Button(button_frame,text="Reset",command=self.reset)
        self.button_reset.pack(side=tk.RIGHT,padx=5)

    def get_live_from_lattice(self):
        try:
            return self.lattice.get_alive()
        except Exception:
            print('lattice not initialised.')
        
    def get_dead_from_lattice(self):
        try:
            return self.lattice.get_dead()
        except Exception:
            print('lattice not initialised.')
        
    def play(self):
        self.loop = True
        curr_population = self.get_live_cells()

        if hasattr(self,"lattice"):
            self.lattice.clear()
        else:
            self.create_lattice(self.lattice_type)

        self.init_lattice(curr_population)
        self.run()
    
    def pause(self):
        self.loop = False

    def next(self):
        curr_population = self.get_live_cells()

        if hasattr(self,"lattice"):
            self.lattice.clear()
        else:
            self.create_lattice(self.lattice_type)

        self.init_lattice(curr_population)
        self.next_gen()
        self.live_cells = self.get_live_from_lattice()


    def random(self):
        if not hasattr(self,"lattice"):
            self.create_lattice(self.lattice_type)
        
        self.lattice.rnd_population()
        live = self.lattice.get_alive()
        self.live_cells = live
        dead = self.lattice.get_dead()
        self.paint(live,dead)
    
    def reset(self):
        self.lattice.clear()
        self.live_cells = []
        self.gens = 0
        for cell_id, (i, j) in self.cells_by_id.items():
            self.canvas.itemconfig(cell_id, fill=self.C_DEAD)
        self.gen_label.config(text=f"generation: {self.gens}")

    def paint(self):
        pass

    def next_gen(self):
        self.lattice.update()
        live = self.lattice.get_alive()
        dead = self.lattice.get_dead()

        # update colors
        self.paint(live,dead)

        # display count of gens
        self.gens += 1
        self.gen_label.config(text=f"generation: {self.gens}")

    def run(self):
        self.next_gen()
        if self.loop:
            self.canvas.after(self.update_freq, self.run)
        else: # if paused, update live cells
            self.live_cells = self.get_live_from_lattice() 
    
    def get_live_cells(self):
        return self.live_cells
    
    def create_lattice(self,lattice_type: str): # init lattice
        if lattice_type == "square":
            self.lattice = SqLattice((self.ROWS,self.COLS)) # square lattice
        if lattice_type == "hexagonal":
            self.lattice = HexLattice(self.ROWS//2,self.COLS) # shexagonalquare lattice

    def init_lattice(self,population): # init config
        self.lattice.set_states(population,True)


class Square(GUI):
    def __init__(self, row, col, off = 120):
        super().__init__(row, col, off)
        self.lattice_type = "square"
        self.cell_width = self.X//self.COLS
        self.cell_height = self.Y//self.ROWS

        self.create_grid()

        self.root.mainloop()

    def create_grid(self):

        # rectangle in canvas
        for i in range(self.ROWS):
            for j in range(self.COLS): #create_rectangle
                id = self.canvas.create_oval(i*self.cell_width,j*self.cell_height,(i+1)*self.cell_width,(j+1)*self.cell_height, fill=self.C_DEAD,outline=self.C_CELL_OUTLINE)
                self.cells_by_id[id] = (j, i) # rectangle ids -> (col,row)

        self.cells_by_pos = { (b,a):rectangle_id  for rectangle_id, (a,b) in self.cells_by_id.items()} # (col,row) -> rectangle ids

        # event: change color on click
        self.canvas.bind("<Button-1>", self.change_color)

    def change_color(self,event):
        # changing color on click
        for cell_id, (i, j) in self.cells_by_id.items():
            coords = self.canvas.coords(cell_id)
            x1, y1, x2, y2 = coords
            if x1 < event.x < x2 and y1 < event.y < y2:
                # change color of the cell on click
                current_color = self.canvas.itemcget(cell_id, "fill")
                new_color = self.C_LIVE if current_color == self.C_DEAD else self.C_DEAD
                self.canvas.itemconfig(cell_id, fill=new_color)

                # update indices of live cells
                if new_color == self.C_LIVE:
                    self.live_cells.append((i, j)) # (col,row)
                else:
                    try:
                        self.live_cells.remove((i, j)) # (col,row)
                    except Exception as e:
                        print(f'{e} \t {(i,j)}')
                        self.lattice.plot()
                        plt.show()        
                break

    def paint(self,live,dead):
        for (a,b) in live:
            cell_id = self.cells_by_pos[(b,a)] # access via (col,row)
            self.canvas.itemconfig(cell_id, fill=self.C_LIVE)

        for (a,b) in dead:
            cell_id = self.cells_by_pos[(b,a)] # access via (col,row)
            self.canvas.itemconfig(cell_id, fill=self.C_DEAD)


class Hexagonal(GUI):
    def __init__(self, row, col, off = 120):
        super().__init__(row, col, off)

        self.lattice_type = "hexagonal"

        self.size = self.Y/(1.5*self.ROWS) # height single hexagon: distance between centers in consecutive rows = 3/2*size
        self.linalghex = LinAlgHex(self.size)

        self.x_offset = 0.5*np.sqrt(3)*self.size
        self.y_offset = self.size + 3
        
        new_width = np.sqrt(3)*self.size*self.COLS + self.x_offset - 3
        new_height = 1.5*self.size*self.ROWS + 0.5*self.y_offset
        
        self.canvas.config(width=new_width, height=new_height) # resize canvas
        (f'{self.X+self.off}x{self.Y+self.off}')
        self.root.geometry(f"{int(new_width)+self.off}x{int(new_height)+self.off}") # resize window

        self.create_grid()

        self.root.mainloop()
 
    def create_grid(self):
        # construct hex grid!
        rows = self.ROWS//2
        cols = self.COLS 

        # HECS
        even_rows = [ [(0,r,c) for c in range(cols)] for r in range(rows)]
        odd_rows = [ [(1,r,c) for c in range(cols)] for r in range(rows)]

        # centers of hexagons
        self.centers_HECS = {}

        for idx in even_rows + odd_rows:
            for (a,r,c) in idx:
                self.centers_HECS.update({(a,r,c):self.linalghex.HECS_to_Cartesian(np.array([a,r,c]).reshape(-1,1))})

        self.centers_by_pos = {tuple(center):hecs for hecs, center in self.centers_HECS.items()}

        # vertices 
        for center in self.centers_HECS.values():
            vertices = self.linalghex.create_hexagon(center) # list of tuples

            vertices = list(sum(vertices, ())) # simple list  

            hex_id = self.canvas.create_polygon(*vertices,fill=self.C_DEAD,outline=self.C_CELL_OUTLINE)
            self.canvas.move(hex_id, self.x_offset, self.y_offset)
            self.cells_by_id[hex_id] = center 
        
        self.cells_by_pos = {(a,b):hex_id  for hex_id, (a,b) in self.cells_by_id.items()} # (col,row) -> center ids

        self.cells_id_by_HECS = { self.centers_by_pos[(a,b)]: hex_id for hex_id, (a,b) in self.cells_by_id.items() }

        # event: change color on click
        self.canvas.bind("<Button-1>", self.change_color)

        # FOR DEBUGGING / ORIENTATION
        # for idx in self.centers_HECS.keys():
        #     x,y = self.centers_HECS[idx]
        #     tk.Label(text=f"{idx}", bg=self.C_DEAD, font=("Ariel", 10, "italic"), fg="white").place(x=x+2.2*self.x_offset, y=y+self.y_offset)
      
    def change_color(self,event):
        # changing color on click
        p = np.array([event.x - self.x_offset,event.y - self.y_offset])
        
        dmin = np.inf
        cmin = None

        for center in self.centers_HECS.values():
            c = np.array(center)
            d = np.linalg.norm(p - c) 
            if dmin > d:
                dmin = d 
                cmin = center
        
        cell_id = self.cells_by_pos[cmin] 

        current_color = self.canvas.itemcget(cell_id, "fill")
        new_color = self.C_LIVE if current_color == self.C_DEAD else self.C_DEAD
        self.canvas.itemconfig(cell_id, fill=new_color)

        # update indices of live cells
        if new_color == self.C_LIVE:
            self.live_cells.append(self.centers_by_pos[cmin]) # (col,row)
        else:
            try:
                self.live_cells.remove(self.centers_by_pos[cmin]) # (col,row)
            except Exception as e:
                print(f'{e}')   

    def paint(self,live,dead):
        for (a,r,c) in live:
            cell_id = self.cells_id_by_HECS[(a,r,c)] # live
            self.canvas.itemconfig(cell_id, fill=self.C_LIVE)

        for (a,r,c) in dead:
            cell_id = self.cells_id_by_HECS[(a,r,c)] # dead
            self.canvas.itemconfig(cell_id, fill=self.C_DEAD)
