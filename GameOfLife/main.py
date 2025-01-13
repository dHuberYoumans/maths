from game_of_life import *
from matplotlib.animation import FuncAnimation
from gui import *

if __name__ == "__main__":
    # gui = Square(20,20)
    gui = Hexagonal(50,50,X=500,Y=500)
    gui.play()
