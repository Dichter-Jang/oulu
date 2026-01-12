from textwrap import wrap
from typing import Optional

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import mld.data.humanml.utils.paramUtil as paramUtil

skeleton = paramUtil.t2m_kinematic_chain


def plot_3d_motion(
    save_path: str,
    joints: np.ndarray,
    title: str,
    figsize: tuple[int, int] = (3, 3),
    fps: int = 120,
    radius: int = 3,
    kinematic_tree: list = skeleton,
    hint: Optional[np.ndarray] = None
) -> None:
    """
    Save a 3D skeleton animation to mp4.

    Compatibility fixes for matplotlib>=3.8:
    - Do NOT assign to ax.lines / ax.collections (read-only ArtistList)
    - update() returns artists (helps avoid blank frames)
    - Use fig.add_subplot(..., projection='3d') instead of Axes3D(fig)
    - blit=False for 3D animations
    - Use ax.set_axis_off() instead of plt.axis('off')
    """

    title = "\n".join(wrap(title, 20))

    def init_axes():
        ax.set_xlim3d([-radius / 2, radius / 2])
        ax.set_ylim3d([0, radius])
        ax.set_zlim3d([-radius / 3.0, radius * 2 / 3.0])
        fig.suptitle(title, fontsize=10)
        ax.grid(False)
        ax.set_axis_off()
        return []

    def plot_xzPlane(minx, maxx, miny, minz, maxz):
        verts = [
            [minx, miny, minz],
            [minx, miny, maxz],
            [maxx, miny, maxz],
            [maxx, miny, minz],
        ]
        xz_plane = Poly3DCollection([verts])
        xz_plane.set_facecolor((0.5, 0.5, 0.5, 0.5))
        ax.add_collection3d(xz_plane)
        return xz_plane

    # (seq_len, joints_num, 3)
    data = joints.copy().reshape(len(joints), -1, 3)

    # scale for visualization
    data *= 1.3

    if hint is not None:
        mask = hint.sum(-1) != 0
        hint = hint[mask]
        hint *= 1.3

    # Create figure/axis (recommended for new matplotlib)
    fig = plt.figure(figsize=figsize)
    plt.tight_layout()
    ax = fig.add_subplot(111, projection="3d")

    # Pre-compute stats
    MINS = data.min(axis=0).min(axis=0)
    MAXS = data.max(axis=0).max(axis=0)

    colors = [
        "#DD5A37", "#D69E00", "#B75A39", "#DD5A37", "#D69E00",
        "#FF6D00", "#FF6D00", "#FF6D00", "#FF6D00", "#FF6D00",
        "#DDB50E", "#DDB50E", "#DDB50E", "#DDB50E", "#DDB50E",
    ]

    frame_number = data.shape[0]

    # Normalize height
    height_offset = MINS[1]
    data[:, :, 1] -= height_offset
    if hint is not None:
        hint[..., 1] -= height_offset

    # Trajectory based on root joint
    trajec = data[:, 0, [0, 2]]

    # Center motion around root on XZ
    data[..., 0] -= data[:, 0:1, 0]
    data[..., 2] -= data[:, 0:1, 2]

    def clear_artists():
        # Matplotlib>=3.8 returns ArtistList without pop/setter; remove explicitly
        for ln in list(ax.lines):
            ln.remove()
        for col in list(ax.collections):
            col.remove()

    def update(index):
        clear_artists()
        artists = []

        # View settings
        ax.view_init(elev=120, azim=-90)
        # ax.dist is deprecated in some versions but still works in many;
        # keep it, but if it errors later, we can remove it.
        try:
            ax.dist = 7.5
        except Exception:
            pass

        # Ground plane
        plane = plot_xzPlane(
            MINS[0] - trajec[index, 0],
            MAXS[0] - trajec[index, 0],
            0,
            MINS[2] - trajec[index, 1],
            MAXS[2] - trajec[index, 1],
        )
        artists.append(plane)

        # Hint points
        if hint is not None:
            sc = ax.scatter(
                hint[..., 0] - trajec[index, 0],
                hint[..., 1],
                hint[..., 2] - trajec[index, 1],
                color="#80B79A",
            )
            artists.append(sc)

        # Skeleton chains
        for i, (chain, color) in enumerate(zip(kinematic_tree, colors)):
            linewidth = 4.0 if i < 5 else 2.0
            ln = ax.plot3D(
                data[index, chain, 0],
                data[index, chain, 1],
                data[index, chain, 2],
                linewidth=linewidth,
                color=color,
            )[0]
            artists.append(ln)

        ax.set_axis_off()
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])

        return artists

    # Initialize axes once
    init_axes()

    ani = FuncAnimation(
        fig,
        update,
        frames=frame_number,
        init_func=lambda: update(0),
        interval=1000 / fps,
        blit=False,
        repeat=False,
    )

    ani.save(save_path, fps=fps)
    plt.close(fig)
