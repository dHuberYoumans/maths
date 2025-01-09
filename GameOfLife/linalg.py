import numpy as np

# LINEAR ALGEBRA FOR HEXAGONAL LATTICE
class LinAlgHex():
    def __init__(self,size: float):
        self.size = size
        self.T = np.sqrt(3)*size*np.array([[0.5,0,1],[0.5*np.sqrt(3), np.sqrt(3),0]]) # lin trafo: HECS -> Cartesian center

    def HECS_to_Cartesian(self,p:np.ndarray):
        """
        Computes centers of hexagon given by HECS coordinate (a,r,c) via linear transformation (x,y) = T (a,r,c)

        Parameters:
        -----------
        p: np.ndarray, shape (3,1)
            (a,r,c) coordinate 

            Returns:
            --------
            center: tuple[float,float]
                center (x,y) of hexagon
        """
        assert p.shape == (3,1), f"expected shape (3,1), got {p.shape}"

        return tuple((self.T @ p).flatten())

    def create_hexagon(self,center:tuple[float,float]):
        """
        Computes the 6 vertices of a regular hexagon centered at center

        Parameters:
        -----------
        center: tuple[float,float]
            center of the gexagon
        
        Returns:
        --------
        vertices: list
            list of coordinates (given by 2-tuples) of the 6 vertices of the hexagon
        """
        i,j = center 
        
        vertices = [
            (i, j + self.size), 
            (i - 0.5 * np.sqrt(3)*self.size, j + 0.5*self.size), 
            (i - 0.5 * np.sqrt(3)*self.size, j - 0.5*self.size), 
            (i, j - self.size), 
            (i + 0.5 * np.sqrt(3)*self.size, j - 0.5*self.size), 
            (i + 0.5 * np.sqrt(3)*self.size, j + 0.5*self.size), 
            ]
        return vertices

    def get_neighbours(self,hexagon: tuple[int,int,int]):
        """
        Computes the 6 neighbours of a hexagon within the HECS

        Parameters:
        -----------
        hexagon: tuple[int,int,int]
            center hexagon in HECS representation (a,r,c)
        
        Returns:
        --------
        neighbours: tuple
            tuple of the 6 nieghbors, each in HECS represenation in the order (right, right above, left above, left, left below, right below):
            3 2
            4     1
            5 6
        """
        a,r,c = hexagon

        right = (a, r, c + 1)
        right_above = (1 - a, r - (1 - a), c + a)
        left_above = (1 - a, r - (1 - a), c - (1 - a))
        left = (a, r, c - 1)
        left_below = (1 - a, r + a, c - (1 - a))
        right_below = (1 - a, r + a, c + a)

        return (right, right_above, left_above, left, left_below, right_below)


        