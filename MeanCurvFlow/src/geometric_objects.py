import os
import sys
from pathlib import Path

cwd = Path(os.getcwd())
sys.path.append(str(cwd.parent)+'/src/')

import numpy as np
import matplotlib.pyplot as plt

from sklearn.neighbors import NearestNeighbors

import geometry as geom

import numpy.typing as npt
from typing import List,Tuple


Vector = npt.NDArray[np.float64]
Vertices = List[Vector]
Edge = Tuple[int,int]
Face = Tuple[int,int,int]

class GeometricObject3D():
    def __init__(self) -> None:
        self.surf = (None,None,None)

    def get_vertices(self,as_array:bool = False) -> list[Vector] | np.ndarray[Vector]:
        if as_array:
            return np.array(self.surf[0])
        else:
            return self.surf[0]
        
    def get_edges(self) -> set[Edge]:
        return self.surf[1]
    
    def get_faces(self) -> set[Face]:
        return self.surf[2]
    
    def get_normal(self,v) -> Vector:
        # neighbours = self.get_neighbours(v)
        vertices_ = self.get_vertices(as_array=True)

        nbrs = NearestNeighbors(n_neighbors=10, algorithm='ball_tree').fit(vertices_)
        _, indices = nbrs.kneighbors([v])

        neighbours = vertices_[indices.flatten()]

        normal_, _, _ = geom.pca_analysis(v, neighbours)

        return normal_
    
    def euler_char(self) -> int:
        # DOES NOT YET WORK SINCE SO FAR WE ONLY DEFINE TRIG-MESHES NOT PROPER TRIANGULATIONS
        """
        computes the Euler characteristic of the surface

        :returns: Euler characteristic chi = V - E + F
        :rtype: int
        """
        V_ = len(self.get_vertices())
        E_ = len(self.get_edges())
        F_ = len(self.get_faces())

        return V_ - E_ + F_
    
    def plot_surface(self,figsize=(10,6),v_color='blue',title=None) -> None:
            vertices_ = self.get_vertices(as_array=True)

            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(projection='3d')

            ax.scatter(vertices_[:, 0], vertices_[:, 1], vertices_[:, 2], color=v_color)

            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('z')

            if title:
                plt.title(title)
        
            plt.show()

    def plot_normals(self,figsize=(10,6),v_color='blue',arr_color='black',scale:float=1.,title=None,mesh:bool = False, mesh_color:str = 'gray') -> None:

        vertices_ = self.get_vertices(as_array=True)

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(projection='3d')

        ax.scatter(vertices_[:,0], vertices_[:, 1], vertices_[:, 2], color=v_color)

        if mesh:
            edges_ = self.get_edges()
            for edge in edges_:
                v0 = vertices_[edge[0]]
                v1 = vertices_[edge[1]]
                v = np.array([v0,v1])

                ax.plot(v[:,0],v[:,1],v[:,2],color=mesh_color)

        for v in vertices_:
            # PLOT NORMAL AS ARROW 
            normal_ = scale*self.get_normal(v)
            ax.quiver(
                v[0], v[1], v[2], # START POINT OF VECTOR
                normal_[0] , normal_[1], normal_[2], # DIRECTION
                color = arr_color, alpha = 0.8, lw = 1,
            )


        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')

        if title:
            plt.title(title)

        plt.show()

    def plot_mesh(self,figsize=(10,6),v_color='blue',e_color='black',title=None) -> None:
        vertices_ = self.get_vertices(as_array=True)
        edges_ = self.get_edges()

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(projection='3d')

        ax.scatter(vertices_[:, 0], vertices_[:, 1], vertices_[:, 2], color=v_color)

        for edge in edges_:
            v0 = vertices_[edge[0]]
            v1 = vertices_[edge[1]]
            v = np.array([v0,v1])

            ax.plot(v[:,0],v[:,1],v[:,2],color=e_color)

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')

        if title:
            plt.title(title)
    
        plt.show()


class Icosphere(GeometricObject3D):
    
    def __init__(self,n_subdivision:int = None) -> None:

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

        edges_, faces_ = geom.edges_and_faces(vertices_)

        self.surf = (vertices_, edges_, faces_)

        if n_subdivision:
            self.subdivision(n_subdivision)
            # for n in range(n_subdivision):
            #     self.subdivision()

    def get_subdivision_pts(self,vertices:Vertices,face:Face,n:int=1) -> list[int]:
        idx0, idx1, idx2 = face
        v0, v1, v2 = vertices[idx0], vertices[idx1], vertices[idx2]
        is_old = False

        vertices_idx = []

        # FIND VERTICES
        for i in range(0,n+1):
            # GET POINT ON EDGE v0-v1 AND DEFINE LINE TO CORRESPONDING POINT ON EDGE v0-v2
            p1 = v0 + (i / n) * (v1 - v0) 
            p2 = v0 + (i / n) * (v2 - v0) 

            # ith LINE HAS TWO POINTS ON THE OUTER EDGES AND i POINTS ON THE LINE CONNECTING THOSE POINTS
            for j in range(i+1):
                if i > 0:
                    p = p1 + (j/i)*(p2-p1)
                else:
                    p = p1
                
                # CHECK IF ALREADY IN VERTICES
                for idx, v in enumerate(vertices):
                    if np.allclose(p, v, atol=1e-8): # CHECKS (UP TO TOLERANCE IF TWO ELEMENTS ARE THE SAME)
                        vertices_idx.append(idx) # IF p ALREADY IN vertices, RETURN ITS INDEX
                        is_old = True
                    
                if not is_old:
                    # IF p IS NOT IN vertices, NORMALISE (TO LIE ON SPHERE), ADD IT (UNIT NORM) AND RETURN ITS INDEX
                    vertices.append(p/np.linalg.norm(p)) 
                    vertices_idx.append(len(vertices)-1)
                
                is_old = False

        return vertices_idx
       
    def get_subdivision_edges_and_faces(self,vertices_idx:list[int]) -> tuple[set[Edge], set[Face]]:
        lines = []
        vertices_ = vertices_idx.copy()
        edges_ = set()
        faces_ = set()
        n = int(1/2 * (-1 + np.sqrt(8 *len(vertices_) + 1)) - 1)  # INVERSE TRIANGULAR NUMBER, ZERO-BASED
        for k in range(1,n+2):
            lines.append(vertices_[:k])
            del vertices_[:k]

        for k in range(len(lines)-1):
            for i in range(k+1):
                e1 = tuple(sorted([lines[k][i],lines[k+1][i]]))
                e2 = tuple(sorted([lines[k][i],lines[k+1][i+1]]))
                e3 = tuple(sorted([lines[k+1][i],lines[k+1][i+1]]))

                f = {tuple(sorted([lines[k][i],lines[k+1][i],lines[k+1][i+1]]))}
                if (k > 0) & (i < k):
                    f.update({tuple(sorted([lines[k][i],lines[k][i+1],lines[k+1][i+1]]))})
                

                edges_.update({e1,e2,e3})
                faces_.update(f)
                
        return edges_, faces_

    def subdivision(self,n:int) -> None:
        vertices_ = self.surf[0]
        _ = self.surf[1]
        faces = self.surf[2]

        faces_ = set()
        edges_ = set()

        for face in faces:
            v0, v1, v2 = vertices_[face[0]], vertices_[face[1]], vertices_[face[2]]

            vertices_idx_ = self.get_subdivision_pts(vertices_,face,n)
            new_edges_, new_faces_ = self.get_subdivision_edges_and_faces(vertices_idx_)
            edges_.update(new_edges_)
            faces_.update(new_faces_)

        self.surf = (vertices_, edges_, faces_)
    
    def get_idx(self,v) -> int:
        vertices_ = self.get_vertices()
        for i, u in enumerate(vertices_):
            if np.array_equal(u, v):  
                return i
        return None  # RETURN None IF NOT FOUND
    
    def get_neighbours(self,v) -> np.ndarray[Vector]:
        vertices_ = self.get_vertices()
        edges_ = self.get_edges() 
        v_idx = self.get_idx(v)
        
        neighbours = np.array([ vertices_[nbr_idx] for u_idx, nbr_idx in edges_ if u_idx == v_idx])

        return neighbours


class Cylinder(GeometricObject3D):
     
    def __init__(self,steps:int = 10, R:float = 1.0):

        # UNIFORM RECTANGULAR GRID OF [0,2PI] X [0,1] 
        steps_ = steps
        phi_steps_ = int(np.ceil(2*np.pi*R*steps_))

        z_ = np.linspace(0, 1, steps_+1)
        phi_ = np.linspace(0, 2 * np.pi, phi_steps_+1)
        cols = len(phi_)
        rows = len(z_)

        phi_,z_  = np.meshgrid(phi_,z_)

        self.vertices_on_plane = np.c_[z_.ravel(),phi_.ravel()]

        # MAP TO CYLINDER
        X_, Y_, Z_ = R*np.cos(self.vertices_on_plane[:,1]), R*np.sin(self.vertices_on_plane[:,1]), R*self.vertices_on_plane[:,0] 

        vertices_ = list(np.c_[X_,Y_,Z_])

        # EDGES
        row_idx = np.arange(rows).repeat(cols)      # [0,0,0,...,1,1,1,...,2,2,2,...,row,row,row]
        col_idx = np.tile(np.arange(cols), rows)    # [0,1,2,...,0,1,2,...,0,1,2,...,...]


        # HORIZONTAL EDGES
        # row_idx*cols = which row in which col => + col_idx = which element in that col
        # for horizontal edges: since phi (varying along col) is angular coordinate, it wraps -> (col_idx + 1) % cols
        horizontal_edges = np.c_[row_idx*cols + col_idx, row_idx*cols + (col_idx + 1) % cols ] 

        # VERTICAL EDGES
        # row_idx[:-cols] -> considers all row indices up to the last one: [0,0,0,...,1,1,1,...2,2,2,...,row-1,row-1,row-1]
        vertical_edges = np.c_[row_idx[:-cols] * cols + col_idx[:-cols], (row_idx[:-cols] + 1) * cols + col_idx[:-cols]]

        # DIAGONAL EDGES
        diagonal_edges = np.c_[row_idx[:-cols] * cols + col_idx[:-cols], (row_idx[:-cols] + 1) * cols + (col_idx[:-cols] + 1) % cols]
        valid = col_idx[:-cols] < (cols - 1) # corner case: exclude the last diagonal
        diagonal_edges = diagonal_edges[valid]

        # STACK EDGES
        edges_ = np.vstack([horizontal_edges,vertical_edges,diagonal_edges])
        edges_ = set(map(tuple,edges_)) 


        # FACES 
        # Idea: consider rectangle  (i,i) , (i,i+1), (i+1,i), (i+1,i+1) and divide it into two triangles (faces) by the diagonal edge (i,i) -> (i+1,i+1)
        # lower face = (i,i), (i+1,i), (i+1,i+1)
        # upper face = (i,i), (i,i+1), (i+1,i+1)
        lower_face = np.c_[row_idx[:-cols] * cols + col_idx[:-cols], (row_idx[:-cols] + 1) * cols + col_idx[:-cols], (row_idx[:-cols] + 1) * cols + (col_idx[:-cols] + 1) % cols]
        upper_face = np.c_[row_idx[:-cols] * cols + col_idx[:-cols], row_idx[:-cols] * cols + (col_idx[:-cols] + 1) % cols, (row_idx[:-cols] + 1) * cols + (col_idx[:-cols] + 1) % cols]

        faces_ = np.vstack([lower_face,upper_face]) 
        faces_ = set(map(tuple,faces_))

        # DEFINE SURFACE IN TERMS OF VERTICES, EDGES AND FACES
        self.surf = (vertices_,edges_,faces_)
        
    def get_planar_vertices(self,as_array:bool = False) ->list[Vector] | np.ndarray[Vector]:
        if as_array:
            return self.vertices_on_plane
        else:
            return list(self.vertices_on_plane) 
  
    def plot_planar_mesh(self,figsize:tuple[int,int] = (10,4),title:str = None) -> None:
        # vertices_on_plane = self.get_planar_vertices()
        edges_ = self.get_edges()

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot()

        for v in self.vertices_on_plane:
            ax.scatter(v[0],v[1],color='k')

        for edge in edges_:
            v0 = self.vertices_on_plane[edge[0]]
            v1 = self.vertices_on_plane[edge[1]]
            v = np.array([v0,v1])

            ax.plot(v[:,0],v[:,1],color='gray')

        plt.title(title)
        plt.show()

     