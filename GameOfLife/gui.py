from game_of_life import *
from linalg import *
import tkinter as tk
import time



class GUI():
    def __init__(self, rows: int, cols: int, X: int = 400, Y: int = 300, off: int = 120):
        # lattice type
        self.lattice_type = None

        # grid size
        self.ROWS = rows
        self.COLS = cols
        
        # window 
        self.X = X
        self.Y = Y
        self.off = off
        self.root = tk.Tk()
        self.root.title("Conway's Game of Life")
        self.root.geometry(f'{self.X+self.off}x{self.Y+self.off}')

        # cells
        self.cells = {} # cell ids <-> coord
        self.live_cells = []
        
        # gens
        self.gens = 0

        # loop
        self.loop = True

        self.create_layout()

        self.grid = None

        # self.root.mainloop()

    def create_canvas(self):
        # canvas
        self.canvas = tk.Canvas(self.root, width=self.X, height=self.Y, bg="green")
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

    def pause(self):
        self.loop = False
        
    def play(self):
        self.loop = True
        curr_population = self.get_live_cells()

        try:
            self.lattice.clear()
        except:
            self.create_lattice(self.lattice_type)

        self.init_lattice(curr_population)
        self.next_gen()
    
    def reset(self):
        self.lattice.clear()
        self.live_cells = []
        self.gens = 0
        for cell_id, (i, j) in self.cells.items():
            self.canvas.itemconfig(cell_id, fill="white")
        self.gen_label.config(text=f"generation: {self.gens}")

    def next_gen(self):
        self.lattice.update()
        live = self.lattice.get_alive()
        dead = self.lattice.get_dead()

        # update colors
        for (a,b) in live:
            cell_id = self.grid[(b,a)] # access via (col,row)
            self.canvas.itemconfig(cell_id, fill='black')

        for (a,b) in dead:
            cell_id = self.grid[(b,a)] # access via (col,row)
            self.canvas.itemconfig(cell_id, fill='white')

        # display count of gens
        self.gens += 1
        self.gen_label.config(text=f"generation: {self.gens}")

        if self.loop:
            self.canvas.after(200, self.next_gen)
        else: # if paused (self.loop = False), update live cells
            self.live_cells = self.get_live_from_lattice() 
    
    def get_live_cells(self):
        return self.live_cells
    
    def create_lattice(self,lattice_type: str): # init lattice
        if lattice_type == "square":
            self.lattice = SqLattice((self.ROWS,self.COLS)) # square lattice
        # if lattice_type == "hexagonal":
        #     self.lattice = HexLattice(self.ROWS) # shexagonalquare lattice

    def init_lattice(self,population): # init config
        self.lattice.set_state(population,True)


class Square(GUI):
    def __init__(self, row, col, X = 400, Y = 300, off = 120):
        super().__init__(row, col, X, Y, off)
        self.lattice_type = "square"
        self.cell_width = X//self.COLS
        self.cell_height = Y//self.ROWS

        self.create_grid()

        self.root.mainloop()

    def create_grid(self):

        # rectangle in canvas
        for i in range(self.ROWS):
            for j in range(self.COLS):
                id = self.canvas.create_rectangle(i*self.cell_width,j*self.cell_height,(i+1)*self.cell_width,(j+1)*self.cell_height, fill="white",outline="black")
                self.cells[id] = (j, i) # rectangle ids -> (col,row)

        self.grid = { (b,a):rectangle_id  for rectangle_id, (a,b) in self.cells.items()} # (col,row) -> rectangle ids

        # event: change color on click
        self.canvas.bind("<Button-1>", self.change_color)

    def change_color(self,event):
        # changing color on click
        for cell_id, (i, j) in self.cells.items():
            coords = self.canvas.coords(cell_id)
            x1, y1, x2, y2 = coords
            if x1 < event.x < x2 and y1 < event.y < y2:
                # change color of the cell on click
                current_color = self.canvas.itemcget(cell_id, "fill")
                new_color = "black" if current_color == "white" else "white"
                self.canvas.itemconfig(cell_id, fill=new_color)

                # update indices of live cells
                if new_color == "black":
                    self.live_cells.append((i, j)) # (col,row)
                else:
                    try:
                        self.live_cells.remove((i, j)) # (col,row)
                    except Exception as e:
                        print(f'{e} \t {(i,j)}')
                        self.lattice.plot()
                        plt.show()        
                break


class Hexagonal(GUI):
    def __init__(self, row, col, X = 400, Y = 300, off = 120):
        super().__init__(row, col, X, Y, off)

        self.lattice_type = "hexagonal"

        self.size = self.Y/(1.5*self.ROWS) # height single hexagon: distance between centers in consecutive rows = 3/2*size
        self.linalghex = LinAlgHex(self.size)

        self.x_offset = 0.5*np.sqrt(3)*self.size
        self.y_offset = self.size + 3
        
        new_width = np.sqrt(3)*self.size*self.COLS + self.x_offset - 3
        new_height = 1.5*self.size*self.ROWS + 0.5*self.y_offset

        print(f'{new_width = }')
        print(f'{new_height = }')
        
        self.canvas.config(width=new_width, height=new_height) # resize canvas
        (f'{self.X+self.off}x{self.Y+self.off}')
        self.root.geometry(f"{int(new_width)+self.off}x{int(new_height)+self.off}") # resize window


        self.create_grid()

        self.root.mainloop()

    
    def create_grid(self):
        # construct hex grid!
        rows = self.ROWS//2
        cols = self.COLS #int(self.X//(np.sqrt(3)*self.size)) # width single hexagon: sqrt(3)*size

        # HECS
        even_rows = np.array([ [(0,r,c) for c  in range(cols)] for r in range(rows)])
        odd_rows = np.array([ [(1,r,c) for c  in range(cols)] for r in range(rows)])

        # centers of hexagons
        centers = []

        for a in [even_rows,odd_rows]:
            for row in a:
                for c in row:
                    centers.append(self.linalghex.HECS_to_Cartesian(c.reshape(-1,1)))

        # vertices 
        for center in centers:
            vertices = self.linalghex.create_hexagon(center) # list of tuples

            vertices = list(sum(vertices, ())) # simple list  

            hex_id = self.canvas.create_polygon(*vertices,fill="white",outline="black")
            self.canvas.move(hex_id, self.x_offset, self.y_offset) # move hexagon
            self.cells[hex_id] = center # save hex_id to center of hexagon
        
        self.grid = {(b,a): hex_id  for hex_id, (a,b) in self.cells.items()} # (col,row) -> center ids

        # event: change color on click
        self.canvas.bind("<Button-1>", self.change_color)
      
