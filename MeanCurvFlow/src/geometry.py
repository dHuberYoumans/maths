import numpy as np

from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
import itertools



def get_midpoint_idx(v1,v2,vertices):
    m = v1 + (v2 - v1) / 2

    # CHECK IF ALREADY IN VERTICES
    for i, v in enumerate(vertices):
        if np.allclose(m, v, atol=1e-8): # CHECKS (UP TO TOLERANCE IF TWO ELEMENTS ARE THE SAME)
            return i  # IF m ALREADY IN vertices, RETURN ITS INDEX
        
    # IF m IS NOT IN vertices, ADD IT AND RETURN ITS INDEX
    vertices.append(m)

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

def EF(vertices,n_nbrs:int = 6):
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
