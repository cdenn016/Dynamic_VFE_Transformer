"""
Gauge Transformer Architecture Diagrams
========================================

Publication-quality figures showing:
1. Gauge-VFE Transformer (variational free energy descent)
2. Gauge-Hamiltonian Transformer (symplectic dynamics)

Author: Generated for publication
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle
from matplotlib.lines import Line2D
import numpy as np

# Style settings
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['text.usetex'] = False  # Set True if LaTeX available

# Color palette (colorblind-friendly)
COLORS = {
    'embedding': '#4477AA',      # Blue
    'attention': '#EE6677',      # Red/pink
    'ffn_vfe': '#228833',        # Green
    'ffn_ham': '#CCBB44',        # Yellow
    'output': '#66CCEE',         # Cyan
    'math': '#AA3377',           # Purple
    'arrow': '#333333',          # Dark gray
    'bg_light': '#F5F5F5',       # Light gray background
    'manifold': '#BBBBBB',       # Gray for manifold
}


def draw_rounded_box(ax, x, y, width, height, label, color, fontsize=9,
                     text_color='white', alpha=0.9):
    """Draw a rounded rectangle with centered text."""
    box = FancyBboxPatch(
        (x - width/2, y - height/2), width, height,
        boxstyle="round,pad=0.02,rounding_size=0.1",
        facecolor=color, edgecolor='black', linewidth=1.2, alpha=alpha,
        transform=ax.transData, zorder=2
    )
    ax.add_patch(box)
    ax.text(x, y, label, ha='center', va='center', fontsize=fontsize,
            color=text_color, fontweight='bold', zorder=3)
    return box


def draw_arrow(ax, start, end, color='#333333', style='->', connectionstyle='arc3,rad=0'):
    """Draw an arrow between two points."""
    arrow = FancyArrowPatch(
        start, end,
        arrowstyle=style,
        connectionstyle=connectionstyle,
        color=color,
        linewidth=1.5,
        mutation_scale=12,
        zorder=1
    )
    ax.add_patch(arrow)
    return arrow


def draw_math_box(ax, x, y, text, fontsize=8):
    """Draw a math equation in a light box."""
    ax.text(x, y, text, ha='center', va='center', fontsize=fontsize,
            color=COLORS['math'], style='italic',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                     edgecolor=COLORS['math'], alpha=0.8, linewidth=0.8),
            zorder=3)


def create_vfe_architecture(ax):
    """Create the Gauge-VFE architecture diagram."""
    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(-1.5, 8.5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('(a) Gauge-VFE Transformer', fontsize=12, fontweight='bold', pad=10)

    # Input layer - wider box, smaller font
    draw_rounded_box(ax, 1.5, 7.5, 3.0, 0.7, 'Token Embed', COLORS['embedding'], fontsize=8)
    # Move state annotation to the right of the box
    ax.text(3.3, 7.5, r'$(\mu_p, \Sigma_p, \phi_p)$', ha='left', va='center',
            fontsize=8, color=COLORS['math'])

    # Transformer block (repeated)
    block_y = 4.5

    # Attention with SO(3) notation
    draw_rounded_box(ax, 5, block_y + 1.8, 3.8, 0.8, 'KL-Attention', COLORS['attention'], fontsize=9)
    # Add multi-head = SO(3) generators annotation
    ax.text(7.2, block_y + 1.8, r'$H$ heads', ha='left', va='center', fontsize=7, color='gray')
    ax.text(7.2, block_y + 1.4, r'$= G_1,G_2,G_3$', ha='left', va='center', fontsize=7, color=COLORS['math'])
    draw_math_box(ax, 5, block_y + 0.7, r'$\beta_{ij} = \mathrm{softmax}(-\mathrm{KL}/\kappa)$')

    # VFE FFN
    draw_rounded_box(ax, 5, block_y - 0.8, 3.8, 0.8, 'VFE Descent FFN', COLORS['ffn_vfe'], fontsize=9)
    draw_math_box(ax, 5, block_y - 1.9, r'$\mu \leftarrow \mu - \eta \nabla_\mu F$')

    # Free energy equation
    ax.text(5, block_y - 3.0,
            r'$F = \alpha \mathrm{KL}(q\|p) + \lambda_\beta \sum \beta_{ij} \mathrm{KL} + \mathrm{CE}$',
            ha='center', fontsize=8, color=COLORS['math'], style='italic')

    # Output
    draw_rounded_box(ax, 5, 0.2, 2.5, 0.7, 'Output Proj', COLORS['output'], fontsize=9)
    ax.text(5, -0.6, r'logits $= W_{out} \mu$', ha='center', fontsize=9, color=COLORS['math'])

    # Arrows - adjusted for new positions
    draw_arrow(ax, (1.5, 7.1), (1.5, 6.2))
    ax.annotate('', xy=(3.0, 6.2), xytext=(1.5, 6.2),
                arrowprops=dict(arrowstyle='->', color=COLORS['arrow'], lw=1.5))
    draw_arrow(ax, (3.0, 6.2), (3.0, block_y + 1.8))
    draw_arrow(ax, (5, block_y + 1.4), (5, block_y + 1.1))
    draw_arrow(ax, (5, block_y + 0.3), (5, block_y - 0.4))
    draw_arrow(ax, (5, block_y - 1.2), (5, block_y - 1.5))
    draw_arrow(ax, (5, block_y - 3.4), (5, 0.6))

    # Residual connection
    ax.annotate('', xy=(7.2, block_y - 0.8), xytext=(7.2, block_y + 1.4),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1,
                               connectionstyle='arc3,rad=-0.3', ls='--'))
    ax.text(7.8, block_y + 0.3, '+', fontsize=12, color='gray')

    # Block label - positioned inside top of box
    ax.add_patch(Rectangle((2.7, block_y - 3.3), 4.8, 5.8,
                           fill=False, edgecolor='gray', linestyle='--', linewidth=1))
    ax.text(5, block_y + 2.3, r'$\times L$ layers', ha='center', fontsize=8, color='gray')


def create_hamiltonian_architecture(ax):
    """Create the Gauge-Hamiltonian architecture diagram."""
    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(-1.5, 8.5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('(b) Gauge-Hamiltonian Transformer', fontsize=12, fontweight='bold', pad=10)

    # Input layer - wider box, smaller font
    draw_rounded_box(ax, 1.5, 7.5, 3.0, 0.7, 'Token Embed', COLORS['embedding'], fontsize=8)
    # Move state annotation to the right of the box
    ax.text(3.3, 7.5, r'$(\mu_p, \Sigma_p, \phi_p)$', ha='left', va='center',
            fontsize=8, color=COLORS['math'])

    # Transformer block
    block_y = 4.5

    # Attention with SO(3) notation
    draw_rounded_box(ax, 5, block_y + 1.8, 3.8, 0.8, 'KL-Attention', COLORS['attention'], fontsize=9)
    # Add multi-head = SO(3) generators annotation
    ax.text(7.2, block_y + 1.8, r'$H$ heads', ha='left', va='center', fontsize=7, color='gray')
    ax.text(7.2, block_y + 1.4, r'$= G_1,G_2,G_3$', ha='left', va='center', fontsize=7, color=COLORS['math'])
    draw_math_box(ax, 5, block_y + 0.7, r'$\beta_{ij} = \mathrm{softmax}(-\mathrm{KL}/\kappa)$')

    # Hamiltonian FFN
    draw_rounded_box(ax, 5, block_y - 0.8, 3.8, 0.8, 'Leapfrog FFN', COLORS['ffn_ham'], text_color='black', fontsize=9)

    # Hamiltonian equations - spread out more
    ax.text(5, block_y - 1.9,
            r'$\dot{q} = \partial H/\partial p$,  $\dot{p} = -\partial H/\partial q$',
            ha='center', fontsize=8, color=COLORS['math'], style='italic')

    # Hamiltonian definition
    ax.text(5, block_y - 2.7,
            r'$H = T_\mu + T_\Sigma + T_\phi + F$',
            ha='center', fontsize=8, color=COLORS['math'], style='italic')
    ax.text(5, block_y - 3.2,
            r'(Kinetic)    (Potential)',
            ha='center', fontsize=7, color='gray')

    # Energy conservation badge
    ax.text(8.3, block_y - 0.8, r'$\Delta H \approx 0$', fontsize=8,
            color='white', ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.2', facecolor=COLORS['ffn_ham'],
                     edgecolor='black', linewidth=0.8))

    # Output
    draw_rounded_box(ax, 5, 0.2, 2.5, 0.7, 'Output Proj', COLORS['output'], fontsize=9)
    ax.text(5, -0.6, r'logits $= W_{out} \mu$', ha='center', fontsize=9, color=COLORS['math'])

    # Arrows - adjusted for new positions
    draw_arrow(ax, (1.5, 7.1), (1.5, 6.2))
    ax.annotate('', xy=(3.0, 6.2), xytext=(1.5, 6.2),
                arrowprops=dict(arrowstyle='->', color=COLORS['arrow'], lw=1.5))
    draw_arrow(ax, (3.0, 6.2), (3.0, block_y + 1.8))
    draw_arrow(ax, (5, block_y + 1.4), (5, block_y + 1.1))
    draw_arrow(ax, (5, block_y + 0.3), (5, block_y - 0.4))
    draw_arrow(ax, (5, block_y - 1.2), (5, block_y - 1.5))
    draw_arrow(ax, (5, block_y - 3.5), (5, 0.6))

    # Residual
    ax.annotate('', xy=(7.2, block_y - 0.8), xytext=(7.2, block_y + 1.4),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1,
                               connectionstyle='arc3,rad=-0.3', ls='--'))
    ax.text(7.8, block_y + 0.3, '+', fontsize=12, color='gray')

    # Block label - positioned inside top of box
    ax.add_patch(Rectangle((2.7, block_y - 3.4), 4.8, 5.9,
                           fill=False, edgecolor='gray', linestyle='--', linewidth=1))
    ax.text(5, block_y + 2.3, r'$\times L$ layers', ha='center', fontsize=8, color='gray')


def create_manifold_diagram(ax):
    """Create the belief manifold visualization as a wavy rectangle."""
    ax.set_xlim(-1, 13)
    ax.set_ylim(-0.5, 5.5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('(c) Belief Dynamics on Statistical Manifold', fontsize=12, fontweight='bold', pad=10)

    # Draw wavy rectangle manifold
    # Top edge (wavy)
    x_top = np.linspace(0.5, 11.5, 200)
    y_top = 4.5 + 0.15 * np.sin(4 * np.pi * (x_top - 0.5) / 11)
    # Bottom edge (wavy)
    x_bot = np.linspace(0.5, 11.5, 200)
    y_bot = 0.8 + 0.15 * np.sin(4 * np.pi * (x_bot - 0.5) / 11 + 0.5)

    # Fill the manifold region
    ax.fill_between(x_top, y_bot, y_top, color=COLORS['manifold'], alpha=0.2)
    ax.plot(x_top, y_top, color=COLORS['manifold'], linewidth=2)
    ax.plot(x_bot, y_bot, color=COLORS['manifold'], linewidth=2)
    # Left and right edges
    ax.plot([0.5, 0.5], [y_bot[0], y_top[0]], color=COLORS['manifold'], linewidth=2)
    ax.plot([11.5, 11.5], [y_bot[-1], y_top[-1]], color=COLORS['manifold'], linewidth=2)

    # Manifold label at top
    ax.text(6, 4.9, r'Statistical Manifold $\mathcal{M}$ (SPD matrices)', ha='center', fontsize=10,
            color=COLORS['manifold'], style='italic')

    # Prior point (left side)
    prior_x, prior_y = 2.5, 2.6
    ax.plot(prior_x, prior_y, 'o', color=COLORS['embedding'], markersize=14, zorder=5)
    ax.text(prior_x - 0.8, prior_y, r'Prior $p$', ha='center', va='center', fontsize=9,
            fontweight='bold', color=COLORS['embedding'])

    # Optimal posterior point (SAME endpoint for both methods)
    post_x, post_y = 9.5, 2.6
    ax.plot(post_x, post_y, '*', color='black', markersize=18, zorder=6)
    ax.text(post_x + 0.8, post_y, r'$q^*$', ha='left', va='center', fontsize=11,
            fontweight='bold', color='black')
    ax.text(post_x, post_y - 0.55, r'optimal posterior', ha='center', fontsize=8, color='gray')

    # VFE gradient descent path (dashed, direct descent - no oscillation)
    t_vfe = np.linspace(0, 1, 40)
    x_vfe = prior_x + (post_x - prior_x) * t_vfe
    y_vfe = np.full_like(x_vfe, prior_y)  # Straight horizontal line (direct descent)
    ax.plot(x_vfe, y_vfe, '--', color=COLORS['ffn_vfe'], linewidth=2.5, zorder=4)
    ax.annotate('', xy=(post_x - 0.15, post_y),
                xytext=(x_vfe[-3], y_vfe[-3]),
                arrowprops=dict(arrowstyle='->', color=COLORS['ffn_vfe'], lw=2.5))

    # Hamiltonian orbit path (solid, SPIRAL into endpoint)
    # Use parametric spiral that converges to (post_x, post_y)
    t_ham = np.linspace(0, 1, 150)
    # Spiral parameters: starts at prior, spirals into posterior
    # Radius decreases as we approach the end
    radius = 0.9 * (1 - t_ham)**0.7  # Decreasing radius
    angle = 6 * np.pi * t_ham  # Multiple rotations
    # Center moves from prior to posterior
    center_x = prior_x + (post_x - prior_x) * t_ham
    center_y = prior_y + (post_y - prior_y) * t_ham
    # Add spiral offset
    x_ham = center_x + radius * np.cos(angle)
    y_ham = center_y + radius * np.sin(angle)
    ax.plot(x_ham, y_ham, '-', color='#AA8800', linewidth=2, zorder=4)
    ax.annotate('', xy=(post_x, post_y - 0.1),
                xytext=(x_ham[-5], y_ham[-5]),
                arrowprops=dict(arrowstyle='->', color='#AA8800', lw=2))

    # Add annotation for VFE path
    ax.text(6, 2.95, r'VFE: $-\nabla_\mu F$', fontsize=9, color=COLORS['ffn_vfe'],
            ha='center', style='italic')

    # Add annotation for Hamiltonian path
    ax.text(4.5, 3.6, r'Ham: spiral in', fontsize=9, color='#AA8800',
            ha='center', style='italic')

    # Legend (repositioned)
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['embedding'],
               markersize=10, label=r'Prior $p$ (from embedding)'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor='black',
               markersize=12, label=r'Optimal posterior $q^*$'),
        Line2D([0], [0], linestyle='--', color=COLORS['ffn_vfe'], lw=2.5,
               label=r'VFE: gradient descent'),
        Line2D([0], [0], linestyle='-', color='#AA8800', lw=2.5,
               label=r'Hamiltonian: energy-conserving'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=8, framealpha=0.95)


def create_full_figure():
    """Create the complete figure with all panels."""
    fig = plt.figure(figsize=(14, 12))

    # Create grid with more space for bottom panel and caption
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 0.7], hspace=0.35, wspace=0.2,
                          top=0.92, bottom=0.12)

    # Top row: architectures
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    # Bottom row: manifold visualization (spans both columns)
    ax3 = fig.add_subplot(gs[1, :])

    # Draw each panel
    create_vfe_architecture(ax1)
    create_hamiltonian_architecture(ax2)
    create_manifold_diagram(ax3)

    # Main title
    fig.suptitle('Gauge Transformer Architectures: VFE vs Hamiltonian Dynamics',
                 fontsize=14, fontweight='bold', y=0.96)

    # Add caption with proper spacing
    caption = (
        "Both architectures use KL-divergence based attention with gauge-equivariant parallel transport.\n"
        "VFE descent minimizes free energy via gradient flow. Hamiltonian dynamics conserves energy via symplectic integration.\n"
        "The self-consistency term KL(q||p) anchors beliefs to embedding priors, enabling gradient flow to learn embeddings."
    )
    fig.text(0.5, 0.03, caption, ha='center', fontsize=9, style='italic',
             wrap=True, color='gray')

    return fig


def main():
    """Generate and save the figure."""
    import os

    fig = create_full_figure()

    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Create output paths relative to script location
    pdf_path = os.path.join(script_dir, 'gauge_transformer_architecture.pdf')
    png_path = os.path.join(script_dir, 'gauge_transformer_architecture.png')

    # Save in multiple formats
    fig.savefig(pdf_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
    fig.savefig(png_path, dpi=300, bbox_inches='tight', pad_inches=0.1)

    print(f"✓ Saved: {pdf_path}")
    print(f"✓ Saved: {png_path}")

    plt.show()


if __name__ == '__main__':
    main()
