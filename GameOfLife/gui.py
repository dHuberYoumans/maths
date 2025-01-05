from game_of_life import *
import tkinter as tk
import time


class GUI():
    def __init__(self, row: int, col: int, X: int = 400, Y: int = 300, off: int = 120):
        # grid size
        self.ROW = row
        self.COL = col

        # window size
        self.X = X
        self.Y = Y
        self.off = off

        self.root = tk.Tk()

        # set geometry
        self.root.geometry(f'{self.X+self.off}x{self.Y+self.off}')

        # canvas
        self.canvas = tk.Canvas(self.root, width=self.X, height=self.Y)
        self.canvas.pack(padx=10,pady=10)

        # store rectangle IDs and indices
        self.rectangles = {} # rectangle ids <-> coord
        self.selected = []

        # rectangle in canvas
        for i in range(self.ROW):
            for j in range(self.COL):
                id = self.canvas.create_rectangle(i*X//self.COL,j*Y//self.COL,(i+1)*X//self.COL,(j+1)*Y//self.ROW, fill='white',outline='black')
                self.rectangles[id] = (i, j) # store rectangle ids

        self.grid = { (a,b):rectangle_id  for rectangle_id, (a,b) in self.rectangles.items()} # coord <-> rectangle ids

        # event: click on rectangle to change color
        self.canvas.bind("<Button-1>", self.change_color)

        # count generations
        self.gens = 0
        self.gen_label = tk.Label(self.root,text=f"generations: {self.gens}")
        self.gen_label.pack()

         # run button
        self.button = tk.Button(self.root,text='Go!',command=self.run)
        self.button.pack(pady=10)

        self.root.mainloop()

    # changing color
    def change_color(self,event):
        for rectangle_id, (i, j) in self.rectangles.items():
            coords = self.canvas.coords(rectangle_id)
            x1, y1, x2, y2 = coords
            if x1 < event.x < x2 and y1 < event.y < y2:
                # change color of the rectangle
                current_color = self.canvas.itemcget(rectangle_id, "fill")
                new_color = "black" if current_color == "white" else "white"
                self.canvas.itemconfig(rectangle_id, fill=new_color)

                # update selected indices
                if new_color == "black":
                    self.selected.append((i, j))
                else:
                    self.selected.remove((i, j))
                break

    def run(self):
        self.create_lattice()
        init_population = self.get_selected()
        self.init_lattice(init_population)

        self.next_gen()

    
    def next_gen(self):

        self.lattice.update()
        live = np.where(self.lattice.get_states() == True)
        dead = np.where(self.lattice.get_states() == False)

        # update colors
        for (a,b) in list(zip(live[0],live[1])):
            rectangle_id = self.grid[(a,b)]
            self.canvas.itemconfig(rectangle_id, fill='black')

        for (a,b) in list(zip(dead[0],dead[1])):
            rectangle_id = self.grid[(a,b)]
            self.canvas.itemconfig(rectangle_id, fill='white')

        # display count of gens
        self.gens += 1
        self.gen_label.config(text=f"generation: {self.gens}")

        self.canvas.after(200, self.next_gen)

            


    def get_selected(self):
        return self.selected
    
    def create_lattice(self): # init lattice
        self.lattice = Lattice(self.ROW) # size = ROW = COL for square lattice

    def init_lattice(self,population): # init config
        self.lattice.set_state(population,True)

