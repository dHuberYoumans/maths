from game_of_life import *
import tkinter as tk
import time


class GUI():
    def __init__(self, row: int, col: int, X: int = 400, Y: int = 300, off: int = 120):
        # grid size
        self.ROW = row
        self.COL = col
        
        # window 
        self.X = X
        self.Y = Y
        self.off = off
        self.root = tk.Tk()
        self.root.geometry(f'{self.X+self.off}x{self.Y+self.off}')

        # cell geometry
        self.cell_width = X//self.COL
        self.cell_height = Y//self.ROW
        
        # gens
        self.gens = 0

        # loop
        self.loop = True

        self.create_layout()
        self.create_grid()

        self.root.mainloop()

    def create_canvas(self):
                # canvas
        self.canvas = tk.Canvas(self.root, width=self.X, height=self.Y)
        self.canvas.pack(padx=10,pady=10)

    def create_layout(self):
        self.create_canvas()

        # display current generation
        self.gen_label = tk.Label(self.root,text=f"generations: {self.gens}")
        self.gen_label.pack()

        # buttons
        button_frame = tk.Frame(self.root)
        button_frame.pack(pady=10)  # Add some vertical padding

        # run button
        self.button_go = tk.Button(button_frame,text='Go!',command=self.run)
        self.button_go.pack(side=tk.LEFT,padx=5)

        # stop buttom
        self.button_stop = tk.Button(button_frame,text='Pause',command=self.pause)
        self.button_stop.pack(side=tk.LEFT,padx=5)

        # continue buttom
        self.button_stop = tk.Button(button_frame,text='Continue',command=self.play)
        self.button_stop.pack(side=tk.RIGHT,padx=5)

    def create_grid(self):
        # store rectangle IDs and indices
        self.rectangles = {} # rectangle ids <-> coord
        self.live_cells = []

        # rectangle in canvas
        for i in range(self.ROW):
            for j in range(self.COL):
                id = self.canvas.create_rectangle(i*self.cell_width,j*self.cell_height,(i+1)*self.cell_width,(j+1)*self.cell_height, fill='white',outline='black')
                self.rectangles[id] = (j, i) # rectangle ids -> (col,row)

        self.grid = { (b,a):rectangle_id  for rectangle_id, (a,b) in self.rectangles.items()} # (col,row) -> rectangle ids

        # event: change color on click
        self.canvas.bind("<Button-1>", self.change_color)

    def change_color(self,event):
        # changing color on click
        for rectangle_id, (i, j) in self.rectangles.items():
            coords = self.canvas.coords(rectangle_id)
            x1, y1, x2, y2 = coords
            if x1 < event.x < x2 and y1 < event.y < y2:
                # change color of the rectangle on click
                current_color = self.canvas.itemcget(rectangle_id, "fill")
                new_color = "black" if current_color == "white" else "white"
                self.canvas.itemconfig(rectangle_id, fill=new_color)

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

    def run(self):
        self.loop = True
        self.create_lattice()
        init_population = self.get_selected()
        self.init_lattice(init_population)

        self.next_gen()

    def pause(self):
        self.loop = False
        
    def play(self):
        self.loop = True
        curr_population = self.get_colored()
        self.lattice.clear()
        self.init_lattice(curr_population)
        self.next_gen()

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
    
    def get_selected(self):
        return self.live_cells
    
    def create_lattice(self): # init lattice
        self.lattice = Lattice(self.ROW) # square lattice

    def init_lattice(self,population): # init config
        self.lattice.set_state(population,True)

