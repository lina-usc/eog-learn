import plotly.graph_objects as go


def plot_head(meshes):
    """Render a 3D head mesh in Plotly.

    Parameters
    ----------
    meshes : list of plotly.graph_objects.Mesh3d
        The mesh traces to display (e.g. scalp, eyes, electrode spheres).

    Returns
    -------
    fig : plotly.graph_objects.Figure
    """
    fig = go.Figure(data=meshes)
    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode='data'
        ),
        showlegend=True
    )
    fig.update_layout(width=800, height=800)
    fig.show()
    return fig
