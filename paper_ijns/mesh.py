from simnibs import read_msh
import plotly.graph_objects as go

import numpy as np
import trimesh


def plotly_sphere(center, radius=1.0, resolution=20, color='red', opacity=0.7, name='sphere'):
    """
    Create a Plotly Mesh3d sphere centered at `center` with the given `radius`.

    Parameters
    ----------
    center : array-like of shape (3,)
        The (x, y, z) center of the sphere.
    radius : float
        Sphere radius.
    resolution : int
        Number of divisions along latitude/longitude (controls mesh density).
    color : str
        Color of the sphere.
    opacity : float
        Transparency between 0 (invisible) and 1 (opaque).
    name : str
        Name for the legend.

    Returns
    -------
    go.Mesh3d
        A Plotly Mesh3d object representing the sphere.
    """
    center = np.array(center, dtype=float)

    # Generate spherical coordinates
    phi, theta = np.mgrid[0:np.pi:complex(resolution), 0:2*np.pi:complex(resolution)]
    x = radius * np.sin(phi) * np.cos(theta) + center[0]
    y = radius * np.sin(phi) * np.sin(theta) + center[1]
    z = radius * np.cos(phi) + center[2]

    # Build triangle faces
    vertices = np.column_stack((x.flatten(), y.flatten(), z.flatten()))
    n_phi, n_theta = x.shape
    faces = []
    for i in range(n_phi - 1):
        for j in range(n_theta - 1):
            p1 = i * n_theta + j
            p2 = p1 + 1
            p3 = p1 + n_theta
            p4 = p3 + 1
            faces.append([p1, p2, p3])
            faces.append([p2, p4, p3])
    faces = np.array(faces)

    sphere_mesh = go.Mesh3d(
        x=vertices[:, 0],
        y=vertices[:, 1],
        z=vertices[:, 2],
        i=faces[:, 0],
        j=faces[:, 1],
        k=faces[:, 2],
        color=color,
        opacity=opacity,
        name=name,
        lighting=dict(ambient=0.4, diffuse=0.6, specular=0.5),
        flatshading=False
    )
    return sphere_mesh


def load_surface_mesh(msh_fname, SURFACE_TAG):
    """Extracts vertices and faces for a given surface tag from a SimNIBS .msh."""
    m = read_msh(msh_fname)

    is_triangle = (m.elm.elm_type == 2)
    has_tag = (m.elm.tag1 == SURFACE_TAG) | (m.elm.tag2 == SURFACE_TAG)
    tri_mask = is_triangle & has_tag

    if not np.any(tri_mask):
        raise RuntimeError(f"No triangle elements found with surface tag {SURFACE_TAG}.")

    tri_node_indices = m.elm.node_number_list[tri_mask, :3].astype(int)
    faces = tri_node_indices - 1  # convert to 0-based indices for numpy

    try:
        vertices = np.asarray(m.nodes.node_coord)
    except Exception:
        vertices = np.asarray(m.nodes)

    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise RuntimeError("Unexpected node array shape; expected Nx3 vertex coordinates.")

    mesh = trimesh.Trimesh(vertices, faces, process=False)
    face_centers = mesh.triangles_center
    face_normals = mesh.face_normals
    centroid = mesh.centroid

    dots = np.einsum('ij,ij->i', face_normals, face_centers - centroid)
    mask_outward = dots > 0

    external = mesh.submesh([mask_outward.nonzero()[0]], only_watertight=False)[0]
    external.remove_unreferenced_vertices()
    external.fix_normals()

    return external.vertices, external.faces


def get_mesh(msh_file, SURFACE_TAG, color='lightcoral', opacity=0.5):

    vertices, faces = load_surface_mesh(msh_file, SURFACE_TAG)

    return go.Mesh3d(
        x=vertices[:, 0],
        y=vertices[:, 1],
        z=vertices[:, 2],
        i=faces[:, 0],
        j=faces[:, 1],
        k=faces[:, 2],
        color=color,
        opacity=opacity
    )


def get_eye_mesh(msh_file, color='royalblue', opacity=1.0):
    SURFACE_TAG = 1006    # 1000 + 6 (eye)
    return get_mesh(msh_file, SURFACE_TAG, color=color, opacity=opacity)


def get_scalp_mesh(msh_file, color='lightcoral', opacity=0.5):
    SURFACE_TAG = 1005    # 1000 + 5 (scalp)
    return get_mesh(msh_file, SURFACE_TAG, color=color, opacity=opacity)
