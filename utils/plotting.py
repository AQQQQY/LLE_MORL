
"""
Plotting utilities for LLE-MORL
========================
scatter_returns(...)     – generic 2-D scatter
scatter_front(...)       – draws points + nondominated hull
heatmap_weights(...)     – neural-network weight heatmap
save_fig(fig, path)      – uniform save (dpi, tight_layout)
"""

from __future__ import annotations
from pathlib import Path
from typing import Sequence
import os.path as osp

import numpy as np
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D 
import pandas as pd
from plotly import express as px
from plotly import graph_objects as go

from utils.common import vec_tag, choose_nondominated_points


def plot_single_policy(episodes_vecs, tag, plot_dir, title, objective_dim, fname=None):
    """
    One scatter per policy.
    For D=2 draw 2-D; D=3 draw 3-D; else only first two dims.
    episodes_vecs : (E, D) array or list of 1-D arrays (mean case), E is the number of episodes, D is the objective dimension.
    """
    vecs = np.asarray(episodes_vecs).reshape(-1, objective_dim)
    if objective_dim == 2 or objective_dim >= 4:
        # 2-D scatter (use first two dims if D>2)
        fig = plt.figure()
        plt.scatter(vecs[:, 0], vecs[:, 1], alpha=0.7)
        plt.xlabel("Reward 0"); plt.ylabel("Reward 1")
        if objective_dim >= 4:
            plt.title(f"{title} (showing R0 vs R1, objective_dim={objective_dim})")
            print(f"[warn] objective_dim={objective_dim}: per-policy plot shows only first 2 dims.")
        else:
            plt.title(title)
        if fname is not None:
            save_fig(fig, osp.join(plot_dir, fname))
        else:           
            save_fig(fig, osp.join(plot_dir, f"{tag}_2d.png"))

    elif objective_dim == 3:
        fig = plt.figure()
        ax: Axes3D = fig.add_subplot(111, projection="3d")
        ax.scatter(vecs[:, 0], vecs[:, 1], vecs[:, 2], alpha=0.7)
        ax.set_xlabel("R0"); ax.set_ylabel("R1"); ax.set_zlabel("R2")
        ax.set_title(tag)
        if fname is not None:
            save_fig(fig, osp.join(plot_dir, fname))
        else:           
            save_fig(fig, osp.join(plot_dir, f"{tag}_3d.png"))


def plot_combined(all_vecs, tags, plot_dir, *, title, objective_dim, fname):
    """
    all_vecs : list of (E, D) arrays **or** list of 1-D arrays (mean case), E is the number of episodes, D is the objective dimension.
    the length of all_vecs should match the length of tags K, K is the number of policies
    tags      : list of strings, names (taged by scalarization weights) for each policy
    Plot all policies in one figure.
    """
    # stack     = np.vstack([v.reshape(-1, objective_dim) for v in all_vecs])
    if objective_dim == 2 or objective_dim >= 4:
        fig = plt.figure()
        cmap = plt.get_cmap("viridis", len(all_vecs))
        offset = 0
        for i, vecs in enumerate(all_vecs):
            vecs = np.asarray(vecs).reshape(-1, objective_dim)
            plt.scatter(vecs[:, 0], vecs[:, 1], alpha=0.6,
                        color=cmap(i), label=tags[i])
            offset += len(vecs)
        plt.xlabel("R0"); plt.ylabel("R1"); plt.title(title)
        plt.legend(fontsize="small", ncol=3, loc="upper center",
                   bbox_to_anchor=(0.5, -0.15))
        plt.subplots_adjust(bottom=0.25)
        save_fig(fig, osp.join(plot_dir, fname))
        if objective_dim >= 4:
            print(f"[warn] D={objective_dim}: combined plot shows R0 vs R1 only.")
    elif objective_dim == 3:
        fig = plt.figure()
        ax: Axes3D = fig.add_subplot(111, projection="3d")
        cmap = plt.get_cmap("viridis", len(all_vecs))
        for i, vecs in enumerate(all_vecs):
            vecs = np.asarray(vecs).reshape(-1, 3)
            ax.scatter(vecs[:, 0], vecs[:, 1], vecs[:, 2],
                       alpha=0.6, color=cmap(i), label=tags[i])
        ax.set_xlabel("R0"); ax.set_ylabel("R1"); ax.set_zlabel("R2")
        ax.set_title(title)
        ax.legend(fontsize="small", ncol=3, loc="upper center",
                  bbox_to_anchor=(0.5, -0.15))
        save_fig(fig, osp.join(plot_dir, fname))


def heatmap_weights(
    W: np.ndarray,
    *,
    title: str | None = None,
    path: Path | None = None,
    cmap: str = "coolwarm",
):
    """Quick weight matrix heatmap."""
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(W, cmap=cmap, aspect="auto")
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if path:
        save_fig(fig, path)
    return fig, ax


# ---------------------------------------------------------------------------
# shared save helper (unify dpi/layout)
# ---------------------------------------------------------------------------
def save_fig(fig: plt.Figure, path: str | Path, *, dpi: int = 300):
    """Save figure with tight layout & consistent dpi."""
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)



def plot_combined_pareto_front(args, all_vecs, base_policy_weights, plot_dir, mode='lle', projection=False):
    """
    Plot combined Pareto front for multiple policies.
    all_vecs : list of (n, D) arrays, n is the number of non-dominated points from one LLE, D is the objective dimension. the length of all_vecs should match the number of base policies K

    mode : str, either 'lle' or 'finetune', determines the title and file names
    projection : bool, if True, adds projections to XY, XZ, YZ planes
    """
    all_points = np.vstack(all_vecs)

    if args.objective_dim == 2 or args.objective_dim >= 4:
        fig = plt.figure()
        cmap = plt.get_cmap("viridis", len(all_vecs))
        offset = 0
        for i, vecs in enumerate(all_vecs):
            vecs = np.asarray(vecs).reshape(-1, args.objective_dim)
            plt.scatter(vecs[:, 0], vecs[:, 1], alpha=0.6,
                        color=cmap(i), label=f"{mode.upper()} from policy {base_policy_weights[i]}")
            offset += len(vecs)
        plt.xlabel("R0"); plt.ylabel("R1")
        plt.legend(fontsize="small", ncol=3, loc="upper center",
                   bbox_to_anchor=(0.5, -0.15))
        plt.subplots_adjust(bottom=0.25)
        # Keep this figure open until the nondominated overlay is saved.
        fig.tight_layout()
        fig.savefig(osp.join(plot_dir, f"pareto_{mode}_combined.png"), dpi=300)

        # Choose non-dominated points on the combined set
        # and plot them with a different color
        all_nondominated, _ = choose_nondominated_points(all_points, args.objective_dim)
        # all_nondominated = np.asarray(all_nondominated).reshape(-1, args.objective_dim)
        plt.scatter(all_nondominated[:, 0], all_nondominated[:, 1],
                    color="blue", marker="*", s=100, label="Nondominated Points")
        plt.legend()
        save_fig(fig, osp.join(plot_dir, f"pareto_{mode}_with_nondominated.png"))
        plt.close(fig)

        # draw Pareto front
        plt.figure()
        plt.scatter(all_nondominated[:, 0], all_nondominated[:, 1],
                    color="blue", marker="*", s=100, label="Nondominated Points")
        plt.xlabel("R0"); plt.ylabel("R1")
        plt.legend(fontsize="small", ncol=3, loc="upper center",
                   bbox_to_anchor=(0.5, -0.15))
        save_fig(plt.gcf(), osp.join(plot_dir, "pareto_front.png"))
        plt.close()

    elif args.objective_dim == 3:
        fig = plt.figure()
        ax: Axes3D = fig.add_subplot(111, projection="3d")
        cmap = plt.get_cmap("viridis", len(all_vecs))
        for i, vecs in enumerate(all_vecs):
            vecs = np.asarray(vecs).reshape(-1, 3)
            ax.scatter(vecs[:, 0], vecs[:, 1], vecs[:, 2],
                       alpha=0.6, color=cmap(i), label=f"{mode.upper()} from policy {base_policy_weights[i]}")
        ax.set_xlabel("R0"); ax.set_ylabel("R1"); ax.set_zlabel("R2")
        ax.legend(fontsize="small", ncol=3, loc="upper center",
                  bbox_to_anchor=(0.5, -0.15))
        save_fig(fig, osp.join(plot_dir, f"pareto_{mode}_3d_combined.png"))

        # Choose non-dominated points on the combined set
        all_nondominated, _ = choose_nondominated_points(all_points, args.objective_dim)
        # all_nondominated = np.asarray(all_nondominated).reshape(-1, args.objective_dim)
        ax.scatter(all_nondominated[:, 0], all_nondominated[:, 1],
                   all_nondominated[:, 2], color="blue", marker="*", s=100, label="Nondominated Points")
        ax.legend()
        save_fig(fig, osp.join(plot_dir, f"pareto_{mode}_3d_with_nondominated.png"))
        plt.close(fig)

        # draw Pareto front
        fig = plt.figure()
        ax: Axes3D = fig.add_subplot(111, projection="3d")
        ax.scatter(all_nondominated[:, 0], all_nondominated[:, 1],
                   all_nondominated[:, 2], color="blue", marker="*", s=100, label="Nondominated Points")
        ax.set_xlabel("R0"); ax.set_ylabel("R1"); ax.set_zlabel("R2")
        ax.legend(fontsize="small", ncol=3, loc="upper center",
                  bbox_to_anchor=(0.5, -0.15))
        save_fig(fig, osp.join(plot_dir, "pareto_front_3d.png"))
        plt.close(fig)      

        # ---------- Optional Plotly interactive ----------
        plot_data = []
        for i, vecs in enumerate(all_vecs):
            df = pd.DataFrame(np.asarray(vecs), columns=["R0", "R1", "R2"])
            df["source"] = f"{mode.upper()} from policy {base_policy_weights[i]}"
            plot_data.append(df)
        df_all = pd.concat(plot_data, ignore_index=True)

        fig_px = px.scatter_3d(df_all, x="R0", y="R1", z="R2", color="source",
                                   opacity=0.7)

        # if projection:
        #     x_min, y_min, z_min = np.min(all_points, axis=0)
        #     offset = 0.05
        #     x_proj = x_min - offset * (np.max(all_points[:,0]) - x_min)
        #     y_proj = y_min - offset * (np.max(all_points[:,1]) - y_min)
        #     z_proj = z_min - offset * (np.max(all_points[:,2]) - z_min)
        #     # Projections to XY plane (z = z_proj)
        #     fig_px.add_trace(go.Scatter3d(
        #             x=df_all["R0"],
        #             y=df_all["R1"],
        #         z=np.full_like(df_all["R2"], z_proj),
        #         mode='markers',
        #         marker=dict(size=3, color='lightgray'),
        #     ))
        #     for xi, yi, zi in zip(df_all["R0"], df_all["R1"], df_all["R2"]):
        #         fig_px.add_trace(go.Scatter3d(
        #                 x=[xi, xi], y=[yi, yi], z=[zi, z_proj],
        #                 mode='lines',
        #                 line=dict(color='lightgray', width=1),
        #                 showlegend=False
        #             ))
        #     # Projections to XZ plane (y = y_proj)
        #     fig_px.add_trace(go.Scatter3d(
        #             x=df_all["R0"],
        #             y=np.full_like(df_all["R1"], y_proj),
        #             z=df_all["R2"],
        #             mode='markers',
        #             marker=dict(size=3, color='lightgray'),
        #         ))
        #     for xi, yi, zi in zip(df_all["R0"], df_all["R1"], df_all["R2"]):
        #         fig_px.add_trace(go.Scatter3d(
        #                 x=[xi, xi], y=[yi, y_proj], z=[zi, zi],
        #                 mode='lines',
        #                 line=dict(color='lightgray', width=1),
        #                 showlegend=False
        #             ))
        #     # Projections to YZ plane (x = x_proj)
        #     fig_px.add_trace(go.Scatter3d(
        #             x=np.full_like(df_all["R0"], x_proj),
        #             y=df_all["R1"],
        #             z=df_all["R2"],
        #             mode='markers',
        #             marker=dict(size=3, color='lightgray'),
        #         ))
        #     for xi, yi, zi in zip(df_all["R0"], df_all["R1"], df_all["R2"]):
        #         fig_px.add_trace(go.Scatter3d(
        #                 x=[xi, x_proj], y=[yi, yi], z=[zi, zi],
        #                 mode='lines',
        #                 line=dict(color='lightgray', width=1),
        #                 showlegend=False
        #             ))

        html_path_0 = osp.join(plot_dir, f"pareto_{mode}_3d_combined.html")
        fig_px.write_html(html_path_0, include_plotlyjs='cdn')

        # Add the non-dominated points to the plot
        fig_px.add_trace(go.Scatter3d(
            x=all_nondominated[:, 0],
            y=all_nondominated[:, 1],
            z=all_nondominated[:, 2],
            mode='markers',
            marker=dict(size=5, color='blue'),
            name='Nondominated Points'
        ))
        html_path_1 = osp.join(plot_dir, f"pareto_{mode}_3d_with_nondominated.html")
        fig_px.write_html(html_path_1, include_plotlyjs='cdn')
        
        
        # Draw the Pareto front
        fig_px_front = px.scatter_3d(
            x=all_nondominated[:, 0],
            y=all_nondominated[:, 1],
            z=all_nondominated[:, 2],
            title="Pareto Front",
            labels={"x": "R0", "y": "R1", "z": "R2"}
        )

        # if projection:
        #     x_min, y_min, z_min = np.min(all_nondominated, axis=0)
        #     offset = 0.05
        #     x_proj = x_min - offset * (np.max(all_nondominated[:,0]) - x_min)
        #     y_proj = y_min - offset * (np.max(all_nondominated[:,1]) - y_min)
        #     z_proj = z_min - offset * (np.max(all_nondominated[:,2]) - z_min)

        #     # projections to XY plane (z = z_proj)
        #     fig_px_front.add_trace(go.Scatter3d(
        #         x=all_nondominated[:, 0],
        #         y=all_nondominated[:, 1],
        #         z=np.full_like(all_nondominated[:, 2], z_proj),
        #         mode='markers',
        #         marker=dict(size=3, color='lightgray'),
        #     ))
        #     for xi, yi, zi in zip(all_nondominated[:, 0], all_nondominated[:, 1], all_nondominated[:, 2]):
        #         fig_px_front.add_trace(go.Scatter3d(
        #             x=[xi, xi], y=[yi, yi], z=[zi, z_proj],
        #             mode='lines',
        #             line=dict(color='lightgray', width=1),
        #             showlegend=False
        #         ))
        #     # projections to XZ plane (y = y_proj)
        #     fig_px_front.add_trace(go.Scatter3d(
        #         x=all_nondominated[:, 0],
        #         y=np.full_like(all_nondominated[:, 1], y_proj),
        #         z=all_nondominated[:, 2],
        #         mode='markers',
        #         marker=dict(size=3, color='lightgray'),
        #     ))
        #     for xi, yi, zi in zip(all_nondominated[:, 0], all_nondominated[:, 1], all_nondominated[:, 2]):
        #         fig_px_front.add_trace(go.Scatter3d(
        #             x=[xi, xi], y=[yi, y_proj], z=[zi, zi],
        #             mode='lines',
        #             line=dict(color='lightgray', width=1),
        #             showlegend=False
        #         ))
        #     # projections to YZ plane (x = x_proj)
        #     fig_px_front.add_trace(go.Scatter3d(
        #         x=np.full_like(all_nondominated[:, 0], x_proj),
        #         y=all_nondominated[:, 1],
        #         z=all_nondominated[:, 2],
        #         mode='markers',
        #         marker=dict(size=3, color='lightgray'),
        #     ))
        #     for xi, yi, zi in zip(all_nondominated[:, 0], all_nondominated[:, 1], all_nondominated[:, 2]):
        #         fig_px_front.add_trace(go.Scatter3d(
        #             x=[xi, x_proj], y=[yi, yi], z=[zi, zi],
        #             mode='lines',
        #             line=dict(color='lightgray', width=1),
        #             showlegend=False
        #         ))
        html_path_2 = osp.join(plot_dir, "pareto_front_3d.html")
        fig_px_front.write_html(html_path_2, include_plotlyjs='cdn')
        

    return all_nondominated
