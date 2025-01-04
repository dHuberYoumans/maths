from game_of_life import *
from matplotlib.animation import FuncAnimation

if __name__ == "__main__":
    N = 20 # lattice size
    lattice = Lattice(N)

    # oscillators
    blinker = [(2,1), (2,2), (2,3)] # N = 5
    toad = [(2,2),(2,3),(2,4),(3,1),(3,2),(3,3)] # N = 6
    beacon = [(1,1),(1,2),(2,1),(2,2),(3,3),(3,4),(4,3),(4,4)] # N = 6

    # spaceships
    glider = [(1,1),(2,2),(3,2),(3,1),(3,0)]

    for idx in blinker:
        lattice.set_state(glider,True)

    # plot setup
    fig, ax = plt.subplots()
    ax.grid(True)
    ax.set_xticks(np.arange(N + 1) - 0.5, minor=False)
    ax.set_yticks(np.arange(N + 1) - 0.5, minor=False)

    ax.set_xticklabels([])
    ax.set_yticklabels([])

    ax.tick_params(which='major', size=0) 

    lattice.plot()

    def animate(frame):
        print(frame)

        lattice.update()  # Update the grid based on the Game of Life rules
        lattice.img.set_data(lattice.get_states())  # Update the plot data

        return [lattice.img]

    ani = FuncAnimation(
        fig,          # The figure to animate
        lattice.animate, # The update function to call at each frame
        frames=100,   # Number of frames (can also be an iterable)
        interval=200,  # Time between frames in milliseconds
        blit=False     # Whether to use blitting for efficiency (only redrawing the parts that change)
    )

    plt.show()