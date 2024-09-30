import os
import sys
from pathlib import Path

cwd = Path(os.getcwd())
sys.path.append(str(cwd.parent)+'/src/')

import numpy as np
import matplotlib.pyplot as plt

from sklearn.neighbors import NearestNeighbors

import geometry as geom





class Icosphere():
    
    def __init__(self,n_subdivision:int = None):

        # ICOSPHERE
        phi__ = (1 + np.sqrt(5)) / 2    # GOLDEN RATIO

        # VERTICES: 
        vertices_ = np.array([])

        for eps1 in [1.,-1.]:
            for eps2 in [1.,-1.]:
                vertices_ = np.append(vertices_,[0,eps1*1,eps2*phi__])
                vertices_ = np.append(vertices_,[eps1*1,eps2*phi__,0])
                vertices_ = np.append(vertices_,[eps2*phi__,0,eps1*1])

        vertices_ = vertices_.reshape(-1,3)

        # NORMALISE vertices_
        vertices_ /= np.linalg.norm(vertices_, axis=1)[:, np.newaxis]

        # CONVERT TO LIST
        vertices_ = list(vertices_)

        edges_, faces_ = geom.EF(vertices_)

        self.surf = (vertices_, edges_, faces_)

        if n_subdivision:
            for n in range(n_subdivision):
                self.subdivision()

    def subdivision(self):
        self.surf = geom.subdivide_triangular_mesh(self.surf)

    def plot_surface(self):
            vertices_ = self.get_vertices()

            fig = plt.figure(figsize=(10,6))
            ax = fig.add_subplot(projection='3d')

            ax.scatter(vertices_[:, 0], vertices_[:, 1], vertices_[:, 2], color='blue')

            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('z')
        
            plt.show()

    def get_vertices(self,as_array:bool = False):
        if as_array:
            return np.array(self.surf[0])
        else:
            return self.surf[0]
    
    def get_edges(self):
        return self.surf[1]
    
    def get_faces(self):
        return self.surf[2]
    
    def get_idx(self,v):
        vertices_ = self.get_vertices()
        for i, u in enumerate(vertices_):
            if np.array_equal(u, v):  
                return i
        return None  # RETURN None IF NOT FOUND
    
    def get_neighbours(self,v):
        vertices_ = self.get_vertices()
        edges_ = self.get_edges() 
        v_idx = self.get_idx(v)
        
        neighbours = np.array([ vertices_[nbr_idx] for u_idx, nbr_idx in edges_ if u_idx == v_idx])

        return neighbours


    def get_normal(self,v):
        # neighbours = self.get_neighbours(v)
        vertices_ = self.get_vertices(as_array=True)

        nbrs = NearestNeighbors(n_neighbors=10, algorithm='ball_tree').fit(vertices_)
        _, indices = nbrs.kneighbors([v])

        neighbours = vertices_[indices.flatten()]

        normal_, _, _ = geom.pca_analysis(v, neighbours)

        return normal_
    
    def plot_normals(self,scale:float=1.):

        vertices_ = self.get_vertices()
        vertices_as_arr_ = np.array(vertices_)

        fig = plt.figure(figsize=(10,6))
        ax = fig.add_subplot(projection='3d')

        ax.scatter(vertices_as_arr_[:,0], vertices_as_arr_[:, 1], vertices_as_arr_[:, 2], color='blue')

        for v in vertices_:
            # PLOT NORMAL AS ARROW 
            normal_ = scale*self.get_normal(v)
            ax.quiver(
                v[0], v[1], v[2], # START POINT OF VECTOR
                normal_[0] , normal_[1], normal_[2], # DIRECTION
                color = 'black', alpha = 0.8, lw = 1,
            )


        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')

        plt.show()


class Cylinder():
     
     def __init__(self,n_points:int=10,R:float=1.0):

        # UNIFORM RECTANGULAR GRID OF [0,2PI] X [0,1] 
        steps_ = n_points
        step_size_ = 1 / steps_

        phi_ = np.arange(0,2*np.pi + step_size_,step_size_)
        z_ = np.arange(0,1 + step_size_,step_size_)

        phi_,z_  = np.meshgrid(phi_,z_)

        # MAP TO CYLINDER
        X_, Y_,Z_ = R*np.cos(phi_), R*np.sin(phi_), R*z_  

        vertices_ = np.array([X_.ravel(),Y_.ravel(),Z_.ravel()]).T
        
     