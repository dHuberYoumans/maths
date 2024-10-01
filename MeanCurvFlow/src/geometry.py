import numpy as np
import numpy.typing as npt
from typing import List,Tuple

from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
import itertools


Vector = npt.NDArray[np.float64]
Vertices = List[Vector]
Edge = Tuple[int,int]
Face = Tuple[int,int,int]

def get_interior_pts_idx(vertices:Vertices,face:Face,n:int=2):
    idx0, idx1, idx2 = face
    v0, v1,v2 = vertices[idx0], vertices[idx1], vertices[idx2]

    vertices_idx = []

    # FIND VERTICES
    for i in range(1, n):
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
                
            # IF p IS NOT IN vertices, NORMALISE (TO LIE ON SPHERE), ADD IT AND RETURN ITS INDEX
            vertices.append(p/np.linalg.norm(p))
            vertices_idx.append(len(vertices)-1)

    return vertices_idx
       
def get_new_edges(new_vertices_idx:list[int]):
    pass

def get_new_faces():
    pass



def get_midpoint_idx(v1,v2,vertices):
    m = v1 + (v2 - v1) / 2

    # CHECK IF ALREADY IN VERTICES
    for i, v in enumerate(vertices):
        if np.allclose(m, v, atol=1e-8): # CHECKS (UP TO TOLERANCE IF TWO ELEMENTS ARE THE SAME)
            return i  # IF m ALREADY IN vertices, RETURN ITS INDEX
        
    # IF m IS NOT IN vertices, ADD IT AND RETURN ITS INDEX
    vertices.append(m/np.linalg.norm(m))

    return len(vertices)-1 

def subdivide_triangular_mesh(surf):
    
    vertices_ = surf[0]
    _ = surf[1]
    faces = surf[2]

    faces_ = set()
    edges_ = set()
    
    for face in faces: # TRIANGULAR FACE
        new_faces = []
        new_edges = []

        v0, v1, v2 = face

        edges_old = [
            (v0,v1),
            (v0,v2),
            (v1,v2)
        ]

        midpoint_idx = []

        for edge in edges_old:
            # DEFINE VERTICES OF EDGE
            v = vertices_[edge[0]]
            u = vertices_[edge[1]]

            # COMPUTE MID POINT, ADD TO vertices AND GET IDX OF MIDPOINT
            midpoint_idx.append(get_midpoint_idx(v,u,vertices_))

        # MIDPOINTS
        m0, m1, m2 = midpoint_idx

        # GATHER NEW EDGES

        new_edges.append(tuple(sorted([v0,m0])))
        new_edges.append(tuple(sorted([v0,m1])))

        new_edges.append(tuple(sorted([v1,m0])))
        new_edges.append(tuple(sorted([v1,m2])))

        new_edges.append(tuple(sorted([v2,m1])))
        new_edges.append(tuple(sorted([v2,m2])))

        new_edges.append(tuple(sorted([m0,m1])))
        new_edges.append(tuple(sorted([m0,m2])))
        new_edges.append(tuple(sorted([m1,m2])))

        edges_ = edges_.union(set(new_edges))

        # GATHER NEW FACES

        new_faces.append(tuple(sorted([v0,m0,m1])))

        new_faces.append(tuple(sorted([v1,m0,m2])))

        new_faces.append(tuple(sorted([v2,m1,m2])))

        new_faces.append(tuple(sorted([m0,m1,m2])))

        faces_ = faces_.union(set(new_faces))

        # NORMALISE VERTICES

        vertices_ = np.array(vertices_) 
        vertices_ /=  np.linalg.norm(vertices_, axis=1)[:, np.newaxis]
        vertices_ = list(vertices_)  # CONVERT BACK TO LIST

    return vertices_, edges_, faces_

def edges_and_faces(vertices,n_nbrs:int = 6):
    """
    vertices = list of np.arrays = pts
    """
    
    # FIND EDGES
    nbrs = NearestNeighbors(n_neighbors=n_nbrs, algorithm='ball_tree').fit(vertices)
    _, indices = nbrs.kneighbors(vertices)

    edges = set()

    for i, neighbours in enumerate(indices):
        for neighbour in neighbours:
            if i < neighbour:
                edges.add((i,neighbour)) # e = (nb vertex,idx neighbour) ORDERED LIST

    faces = set() # FACE = (IDX_1,IDX_2,IDX_3) WHERE IDX_i = INDEX OF i-TH VERTEX OF FACE

    for i, neighbours in enumerate(indices):
        pairs_of_nbrs = list(itertools.combinations(neighbours,2))
        sorted_pairs_of_nbrs = [tuple(sorted(pair)) for pair in pairs_of_nbrs]

        for pair in sorted_pairs_of_nbrs:
            if pair in sorted(list(edges)) and i < pair[0]:
                faces.add(tuple([i]+list(pair)))
        
        faces = set([face for face in faces if len(set(face)) == 3])

    return edges, faces

def pca_analysis(p, neighbors,n_components:int = 3):
    # CENTER NBRS AROUND p
    centered_neighbors = neighbors - p
    
    # FIT PCA 
    pca = PCA(n_components=n_components)
    pca.fit(centered_neighbors)
    
    # NORMAL = LAST COMPONENT
    normal = pca.components_[-1]  # last component corresponds to smallest variance

    # ENSURING NORMAL POINTS OUTWARDS
    if np.dot(normal,p) < 0:
        normal = -1*normal

    e1 = pca.components_[0] # EIGENVEC TO lambda_max
    e2 = pca.components_[1]
    
    # ESTIMATE MEAN CURVATURE

    return normal, e1, e2

def fit_quadratic_surf(x,y,z): # FITS SURFACE z(x,y) = ax**2 + by**2 + c xy + dx + ey + f => b = A  where A = {x**2, y**2, xy, x, y, 1} = (1,5) matrix and v = {a,b,c,d,e,f} = (5,1) matrix 
    A = np.c_[x**2,y**2,x*y,x,y,np.ones_like(x)] # STACKS COMPONENTS (x**2, y**2 etc.) AS VECTORS (VECTORISATION) HORIZONTALLY (APPOSED TO VSTACK) - EACH COLUMN REPRESENTS TERMS x**2, y**2, xy, etc
    sol, _, _, _ = np.linalg.lstsq(A,z,rcond=None) # LEAST SQUARE FIT OF LINEAR EQN A v = b ; rcond = CUT OFF CONDITION

    return sol

def estimate_principal_curvatures(point,neighbors):

    # COMPUTE NORMAL AND TANGENT FRAME
    normal, e1, e2 = pca_analysis(point, neighbors)

    # CENTER NBRS
    nbrs = neighbors - point

    # PROJECT NBRS TO TANGENT PLANE
    x = nbrs @ e1
    y = nbrs @ e2
    z = nbrs @ (-normal) # PROJECTION TO NORMAL - USE -normal SINCE normal POINTS OUTWARD THEN z(x,y) APPROXIMATES SURFACE IN OPPOSITE DIRECTION

    # FIT QUADARTIC SURFACE
    a, b, c, _, _, _ = fit_quadratic_surf(x,y,z)

    # COMPUTE PRINCIPAL CURVATURES
    Hess = np.array([[2*a, c], [c, 2*b]])

    kappa1, kappa2 = np.linalg.eigvals(Hess)

    return kappa1, kappa2

def mean_curv(cloud,k:int=15):
    H_ = []

    nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(cloud)
    _, indices = nbrs.kneighbors(cloud)

    for n in range(indices.shape[0]):
        point = cloud[n] 
        
        neighbours = cloud[indices[n]]

        kappa1, kappa2 = estimate_principal_curvatures(point, neighbours)

        H_.append((point,(kappa1 + kappa2) / 2))

    return H_
     