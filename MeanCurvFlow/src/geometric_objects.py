import os
import sys
from pathlib import Path

cwd = Path(os.getcwd())
sys.path.append(str(cwd.parent)+'/src/')

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

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

    def plot_normals(self,figsize=(10,6),v_color='blue',arr_color='black',scale:float=1.,title=None,mesh:bool = False, mesh_color:str = 'gray', fill:bool = False, fill_color:str = 'yellow') -> None:

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(projection='3d')

        if fill:
            vertices_ = self.get_vertices(as_array=True)

            for face in self.get_faces():
                v0 = vertices_[face[0]]
                v1 = vertices_[face[1]]
                v2 = vertices_[face[2]]
                triangle = [np.array([v0,v1,v2])]

                poly = Poly3DCollection(triangle, facecolors=fill_color, edgecolors='gray', linewidths=1, alpha=0.3)
                ax.add_collection3d(poly)

        else:
            vertices_ = self.get_vertices(as_array=True)

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

    def plot_mesh(self,figsize:tuple[int,int] = (10,6),v_color:str = 'blue', e_color:str = 'black', title:str = None, fill:bool = False, fill_color:str = 'yellow') -> None:

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(projection='3d')

        if fill:
            vertices_ = self.get_vertices(as_array=True)

            for face in self.get_faces():
                v0 = vertices_[face[0]]
                v1 = vertices_[face[1]]
                v2 = vertices_[face[2]]
                triangle = [np.array([v0,v1,v2])]

                poly = Poly3DCollection(triangle, facecolors=fill_color, edgecolors='gray', linewidths=1, alpha=0.3)
                ax.add_collection3d(poly)

        else:
            vertices_ = self.get_vertices(as_array=True)

        edges_ = self.get_edges()

        ax.scatter(vertices_[:, 0], vertices_[:, 1], vertices_[:, 2], color=v_color,zorder=1)

        for edge in edges_:
            v0 = vertices_[edge[0]]
            v1 = vertices_[edge[1]]
            v = np.array([v0,v1])

            ax.plot(v[:,0],v[:,1],v[:,2],color=e_color,zorder=-1)

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')

        if title:
            plt.title(title)

        plt.show()

    def info(self) -> None:
        """
        prints the number of vertices (V), edges (E) and faces (F) of the triangulation of the surface
        """
        print(f'V = {len(self.get_vertices())}')
        print(f'E = {len(self.get_edges())}')
        print(f'F = {len(self.get_faces())}')

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


    def get_subdivision_pts(self,vertices:Vertices,face:Face,n:int=1) -> list[int]:
        idx0, idx1, idx2 = face
        v0, v1, v2 = vertices[idx0], vertices[idx1], vertices[idx2]
        is_new = True

        vertices_idx = []

        # FIND VERTICES
        for i in range(0,n+1):
            # GET POINT ON EDGE v0-v1 AND DEFINE LINE TO CORRESPONDING POINT ON EDGE v0-v2
            p1 = v0 + (i / n) * (v1 - v0) 
            p2 = v0 + (i / n) * (v2 - v0) 

            # ith LINE HAS TWO POINTS ON THE OUTER EDGES AND i POINTS ON THE LINE CONNECTING THOSE POINTS
            
            for j in range(i+1):
                is_new = True
                if i > 0:
                    p = p1 + (j/i)*(p2-p1)
                else:
                    p = p1
                
                # CHECK IF ALREADY IN VERTICES
                for idx, v in enumerate(vertices):
                    if np.allclose(p/np.linalg.norm(p), v, atol=1e-8): # CHECKS (UP TO TOLERANCE IF TWO ELEMENTS ARE THE SAME)
                        vertices_idx.append(idx) # IF p ALREADY IN vertices, RETURN ITS INDEX
                        is_new = False
                        break
                        
                if is_new:
                    # IF p IS NOT IN vertices, NORMALISE (TO LIE ON SPHERE), ADD IT (UNIT NORM) AND RETURN ITS INDEX
                    vertices.append(p/np.linalg.norm(p)) 
                    vertices_idx.append(len(vertices)-1)

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

    def __init__(self,steps:int = 10, radius:float = 1.0, height: float = 1.0):

        self.R = radius

        # UNIFORM RECTANGULAR GRID OF [0,2PI] X [0,1] 

        steps_ = steps
        phi_steps_ = int(np.ceil(2*np.pi*self.R*steps_))

        z_ = np.linspace(0, height, steps_+1)
        z_proper = z_.copy()
        phi_ = np.linspace(0, 2 * np.pi, phi_steps_+1) 
        phi_proper = phi_.copy()[:-1] # due to periodic identification of phi consider only up to last point 

        cols = len(phi_) 
        rows = len(z_)

        phi_, z_  = np.meshgrid(phi_,z_) # for plotting and defining edges and faces
        phi_proper, z_proper  = np.meshgrid(phi_proper,z_proper) # considering periodic identification of phi 

        self.vertices_on_plane_padded = np.c_[phi_.ravel(),z_.ravel()] 
        vertices_on_plane_ = np.c_[phi_proper.ravel(),z_proper.ravel()] 

        # MAP TO CYLINDER
        X_, Y_, Z_ = self.R*np.cos(vertices_on_plane_[:,0]), self.R*np.sin(vertices_on_plane_[:,0]), self.R*vertices_on_plane_[:,1] 

        vertices_ = list(np.c_[X_,Y_,Z_])

        # EDGES
        row_idx = np.arange(rows).repeat(cols-1)          # [0,0,0,...,1,1,1,...,2,2,2,...,row,row,row]
        col_idx = np.tile(np.arange(cols-1), rows)        # [0,1,2,...,0,1,2,...,0,1,2,...,...]
        row_idx_padded = np.arange(rows).repeat(cols)     # [0,0,0,...,1,1,1,...,2,2,2,...,row,row,row]
        col_idx_padded = np.tile(np.arange(cols), rows)
 
        # EDGES & FACES
        # idea: in plane, have grid of (phi,z) points -> m x n = len(phi) x len(z) matrix =>  (m*n,) vector (array) 
        # find indices of horizontal, vertical and diagonal edges, stack and return them as a set of tuples

        # HORIZONTAL EDGES
        # row_idx*cols = z-coordinate (row) , col_idx = phi-coordinate (col) => row_idx*cols + col_idx = index of (phi,z) coordinate
        
        horizontal_edges = np.c_[(row_idx*cols) + col_idx, (row_idx*cols) + (col_idx + 1)]

        # VERTICAL EDGES
        
        vertical_edges = np.c_[row_idx[:-cols + 1] * cols + col_idx[:-cols + 1], (row_idx[:-cols + 1] + 1) * cols + col_idx[:-cols + 1]]
        vertical_edges_padded = np.c_[row_idx_padded[:-cols] * cols + col_idx_padded[:-cols], (row_idx_padded[:-cols] + 1) * cols + col_idx_padded[:-cols]] # padding needed for plotting

        # DIAGONAL EDGES
        diagonal_edges = np.c_[row_idx[:-cols+1] * cols + col_idx[:-cols+1], (row_idx[:-cols+1] + 1) * cols + (col_idx[:-cols+1] + 1)]

        # STACK EDGES
        self.edges_padded = np.vstack([horizontal_edges,vertical_edges_padded,diagonal_edges]) 

        edges_ = np.vstack([horizontal_edges,vertical_edges,diagonal_edges])
        edges_.sort() # sort faces such that (i,j) satisfies i < j
        edges_ = set((map(tuple,edges_)))

        # FACES 
        upper_face = np.c_[row_idx[:-(cols-1)] * cols + col_idx[:-(cols-1)], (row_idx[:-(cols-1)] + 1) * cols + col_idx[:-(cols-1)], (row_idx[:-(cols-1)] + 1) * cols + (col_idx[:-(cols-1)] + 1) ]
        lower_face = np.c_[row_idx[:-(cols-1)] * cols + col_idx[:-(cols-1)], row_idx[:-(cols-1)] * cols + (col_idx[:-(cols-1)] + 1), (row_idx[:-(cols-1)] + 1) * cols + (col_idx[:-(cols-1)] + 1) ]

        faces_ = np.vstack([upper_face,lower_face]) 
        faces_.sort() # sort faces such that (i,j,k) satisfies i < j < k
        faces_ = set(map(tuple,faces_))

        # DEFINE SURFACE IN TERMS OF VERTICES, EDGES AND FACES (TRIANGULATION)
        self.surf = (vertices_,edges_,faces_)
           
    def get_planar_vertices(self,as_array:bool = False) ->list[Vector] | np.ndarray[Vector]:
        if as_array:
            return self.vertices_on_plane
        else:
            return list(self.vertices_on_plane) 
  
    def plot_planar_mesh(self,fill:bool = False, fill_color:str = 'yellow', figsize:tuple[int,int] = (10,4),title:str = None) -> None:

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot()

        for edge in self.edges_padded:
            v0 = self.vertices_on_plane_padded[edge[0]]
            v1 = self.vertices_on_plane_padded[edge[1]]
            v = np.array([v0,v1])

            if (v0[0] % (2*np.pi) < 1e-8) and (v0[0] != 0.0):
                ax.plot(v[:,0],v[:,1],color='gray',ls='--',zorder=-1)
            else:
                ax.plot(v[:,0],v[:,1],color='gray',zorder=-1)
            
        for v in self.vertices_on_plane_padded:
            if (v[0] % (2*np.pi) < 1e-8) and (v[0] != 0.0):
                ax.scatter(v[0],v[1],edgecolor='k',facecolors='w',zorder=1)
            else:
                ax.scatter(v[0],v[1],color='k',zorder=1)

        if fill:
            for face in self.get_faces():
                v0 = self.vertices_on_plane_padded[face[0]]
                v1 = self.vertices_on_plane_padded[face[1]]
                v2 = self.vertices_on_plane_padded[face[2]]
                triangle = np.array([v0,v1,v2])

                ax.fill(triangle[:, 0], triangle[:, 1], color=fill_color, alpha=0.3,zorder=-2)

        plt.title(title)
        plt.show()

    def plot_mesh(self,figsize:tuple[int,int] =(10,6),v_color:str ='blue', e_color:str ='black', title:str = None,fill:bool = False, fill_color:str = 'yellow') -> None:
        cp_ = self.surf[0].copy()
        X_, Y_, Z_ = self.R*np.cos(self.vertices_on_plane_padded[:,0]), self.R*np.sin(self.vertices_on_plane_padded[:,0]), self.R*self.vertices_on_plane_padded[:,1] 
        vertices_ = list(np.c_[X_,Y_,Z_])
        self.surf = (vertices_, *self.surf[1:])
        super().plot_mesh(figsize=figsize,v_color=v_color, e_color=e_color, title=title, fill=fill, fill_color=fill_color)
        self.surf = (cp_, *self.surf[1:])

    def plot_normals(self, figsize:tuple[int,int] = (10,6), v_color:str = 'blue', arr_color:str = 'black', scale:float = 1.0, title:str = None, mesh:bool = False, mesh_color:str = 'gray', fill:bool = False, fill_color:str = 'yellow') -> None:
        cp_ = self.surf[0].copy()
        X_, Y_, Z_ = self.R*np.cos(self.vertices_on_plane_padded[:,0]), self.R*np.sin(self.vertices_on_plane_padded[:,0]), self.R*self.vertices_on_plane_padded[:,1] 
        vertices_ = list(np.c_[X_,Y_,Z_])
        self.surf = (vertices_, *self.surf[1:])
        super().plot_normals(figsize = figsize, v_color = v_color, arr_color = arr_color, scale = scale, title = title, mesh = mesh, mesh_color = mesh_color, fill = fill, fill_color = fill_color)
        self.surf = (cp_, *self.surf[1:])