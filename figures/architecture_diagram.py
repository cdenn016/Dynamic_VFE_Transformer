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
    ax.set_ylim(-1, 8)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('(a) Gauge-VFE Transformer', fontsize=12, fontweight='bold', pad=10)

    # Input layer
    draw_rounded_box(ax, 1.5, 7, 2.5, 0.7, 'Token Embedding', COLORS['embedding'])
    ax.text(1.5, 6.1, r'$\mu_p, \Sigma_p, \phi_p$', ha='center', fontsize=9, color=COLORS['math'])

    # Transformer block (repeated)
    block_y = 4.5

    # Attention
    draw_rounded_box(ax, 5, block_y + 1.5, 3.5, 0.8, 'KL-Attention', COLORS['attention'])
    draw_math_box(ax, 5, block_y + 0.5, r'$\beta_{ij} = \mathrm{softmax}(-\mathrm{KL}(q_i \| \Omega_{ij}[q_j])/\kappa)$')

    # VFE FFN
    draw_rounded_box(ax, 5, block_y - 1.2, 3.5, 0.8, 'VFE Descent FFN', COLORS['ffn_vfe'])
    draw_math_box(ax, 5, block_y - 2.2, r'$\mu \leftarrow \mu - \eta \nabla_\mu F$')

    # Free energy equation
    ax.text(5, block_y - 3.2,
            r'$F = \alpha \cdot \mathrm{KL}(q\|p) + \lambda_\beta \sum_{ij} \beta_{ij} \mathrm{KL}(q_i\|\Omega_{ij}[q_j]) + \mathrm{CE}$',
            ha='center', fontsize=8, color=COLORS['math'], style='italic')

    # Output
    draw_rounded_box(ax, 5, 0.5, 2.5, 0.7, 'Output Proj', COLORS['output'])
    ax.text(5, -0.3, r'logits $= W_{out} \mu$', ha='center', fontsize=9, color=COLORS['math'])

    # Arrows
    draw_arrow(ax, (1.5, 6.6), (1.5, 5.5))
    ax.annotate('', xy=(3.2, 5.5), xytext=(1.5, 5.5),
                arrowprops=dict(arrowstyle='->', color=COLORS['arrow'], lw=1.5))
    draw_arrow(ax, (3.2, 5.5), (3.2, block_y + 1.5))
    draw_arrow(ax, (5, block_y + 1.1), (5, block_y + 0.9))
    draw_arrow(ax, (5, block_y + 0.1), (5, block_y - 0.8))
    draw_arrow(ax, (5, block_y - 1.6), (5, block_y - 1.8))
    draw_arrow(ax, (5, block_y - 2.7), (5, 0.9))

    # Residual connection
    ax.annotate('', xy=(7.5, block_y - 1.2), xytext=(7.5, block_y + 1.5),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1,
                               connectionstyle='arc3,rad=-0.3', ls='--'))
    ax.text(8.2, block_y + 0.2, '+', fontsize=12, color='gray')

    # Block label
    ax.add_patch(Rectangle((2.8, block_y - 2.8), 4.4, 5,
                           fill=False, edgecolor='gray', linestyle='--', linewidth=1))
    ax.text(2.9, block_y + 2.1, '×L layers', fontsize=8, color='gray')


def create_hamiltonian_architecture(ax):
    """Create the Gauge-Hamiltonian architecture diagram."""
    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(-1, 8)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('(b) Gauge-Hamiltonian Transformer', fontsize=12, fontweight='bold', pad=10)

    # Input layer
    draw_rounded_box(ax, 1.5, 7, 2.5, 0.7, 'Token Embedding', COLORS['embedding'])
    ax.text(1.5, 6.1, r'$\mu_p, \Sigma_p, \phi_p$', ha='center', fontsize=9, color=COLORS['math'])

    # Transformer block
    block_y = 4.5

    # Attention (same as VFE)
    draw_rounded_box(ax, 5, block_y + 1.5, 3.5, 0.8, 'KL-Attention', COLORS['attention'])
    draw_math_box(ax, 5, block_y + 0.5, r'$\beta_{ij} = \mathrm{softmax}(-\mathrm{KL}(q_i \| \Omega_{ij}[q_j])/\kappa)$')

    # Hamiltonian FFN
    draw_rounded_box(ax, 5, block_y - 1.2, 3.5, 0.8, 'Leapfrog FFN', COLORS['ffn_ham'], text_color='black')

    # Hamiltonian equations
    ax.text(5, block_y - 2.1,
            r'$\dot{q} = \partial H/\partial p$,  $\dot{p} = -\partial H/\partial q$',
            ha='center', fontsize=8, color=COLORS['math'], style='italic')

    # Hamiltonian definition
    ax.text(5, block_y - 2.8,
            r'$H = \underbrace{T_\mu + T_\Sigma + T_\phi}_{\mathrm{Kinetic}} + \underbrace{F(\mu,\Sigma,\phi)}_{\mathrm{Potential}}$',
            ha='center', fontsize=8, color=COLORS['math'], style='italic')

    # Energy conservation badge
    ax.text(8.5, block_y - 1.2, r'$\Delta H \approx 0$', fontsize=8,
            color='white', ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.2', facecolor=COLORS['ffn_ham'],
                     edgecolor='black', linewidth=0.8))

    # Output
    draw_rounded_box(ax, 5, 0.5, 2.5, 0.7, 'Output Proj', COLORS['output'])
    ax.text(5, -0.3, r'logits $= W_{out} \mu$', ha='center', fontsize=9, color=COLORS['math'])

    # Arrows
    draw_arrow(ax, (1.5, 6.6), (1.5, 5.5))
    ax.annotate('', xy=(3.2, 5.5), xytext=(1.5, 5.5),
                arrowprops=dict(arrowstyle='->', color=COLORS['arrow'], lw=1.5))
    draw_arrow(ax, (3.2, 5.5), (3.2, block_y + 1.5))
    draw_arrow(ax, (5, block_y + 1.1), (5, block_y + 0.9))
    draw_arrow(ax, (5, block_y + 0.1), (5, block_y - 0.8))
    draw_arrow(ax, (5, block_y - 1.6), (5, block_y - 1.8))
    draw_arrow(ax, (5, block_y - 3.3), (5, 0.9))

    # Residual
    ax.annotate('', xy=(7.5, block_y - 1.2), xytext=(7.5, block_y + 1.5),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1,
                               connectionstyle='arc3,rad=-0.3', ls='--'))
    ax.text(8.2, block_y + 0.2, '+', fontsize=12, color='gray')

    # Block label
    ax.add_patch(Rectangle((2.8, block_y - 3.4), 4.4, 5.6,
                           fill=False, edgecolor='gray', linestyle='--', linewidth=1))
    ax.text(2.9, block_y + 2.1, '×L layers', fontsize=8, color='gray')


def create_manifold_diagram(ax):
    """Create the belief manifold visualization."""
    ax.set_xlim(-2, 12)
    ax.set_ylim(-1, 7)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('(c) Belief Dynamics on Statistical Manifold', fontsize=12, fontweight='bold', pad=10)

    # Draw curved manifold surface
    theta = np.linspace(-0.3, np.pi + 0.3, 100)
    r = 4
    x_curve = 5 + r * np.cos(theta)
    y_curve = 3 + 0.5 * r * np.sin(theta)
    ax.fill_between(x_curve, y_curve - 0.3, y_curve + 0.3,
                    color=COLORS['manifold'], alpha=0.3)
    ax.plot(x_curve, y_curve, color=COLORS['manifold'], linewidth=2)

    # Prior point
    prior_x, prior_y = 2, 3.2
    ax.plot(prior_x, prior_y, 'o', color=COLORS['embedding'], markersize=12, zorder=5)
    ax.text(prior_x, prior_y + 0.6, r'Prior $p$', ha='center', fontsize=9, color=COLORS['embedding'])
    ax.text(prior_x, prior_y - 0.5, r'$(\mu_p, \Sigma_p)$', ha='center', fontsize=8, color=COLORS['math'])

    # Posterior point (VFE)
    post_vfe_x, post_vfe_y = 5.5, 3.5
    ax.plot(post_vfe_x, post_vfe_y, 's', color=COLORS['ffn_vfe'], markersize=12, zorder=5)
    ax.text(post_vfe_x + 0.5, post_vfe_y + 0.6, r'Posterior $q$ (VFE)', ha='center', fontsize=9, color=COLORS['ffn_vfe'])

    # Posterior point (Hamiltonian)
    post_ham_x, post_ham_y = 7.5, 2.8
    ax.plot(post_ham_x, post_ham_y, '^', color=COLORS['ffn_ham'], markersize=12, zorder=5)
    ax.text(post_ham_x + 0.8, post_ham_y + 0.6, r'Posterior $q$ (Ham)', ha='center', fontsize=9,
            color='#998800')  # Darker yellow for readability

    # VFE gradient descent path
    t_vfe = np.linspace(0, 1, 20)
    x_vfe = prior_x + (post_vfe_x - prior_x) * t_vfe + 0.3 * np.sin(3 * np.pi * t_vfe)
    y_vfe = prior_y + (post_vfe_y - prior_y) * t_vfe + 0.2 * np.sin(2 * np.pi * t_vfe)
    ax.plot(x_vfe, y_vfe, '--', color=COLORS['ffn_vfe'], linewidth=2, label='VFE descent')
    ax.annotate('', xy=(post_vfe_x - 0.1, post_vfe_y - 0.05),
                xytext=(x_vfe[-2], y_vfe[-2]),
                arrowprops=dict(arrowstyle='->', color=COLORS['ffn_vfe'], lw=2))

    # Hamiltonian orbit path
    t_ham = np.linspace(0, 1, 50)
    x_ham = prior_x + (post_ham_x - prior_x) * t_ham
    y_ham = prior_y + (post_ham_y - prior_y) * t_ham + 0.8 * np.sin(4 * np.pi * t_ham) * (1 - t_ham)
    ax.plot(x_ham, y_ham, '-', color=COLORS['ffn_ham'], linewidth=2, label='Hamiltonian orbit')
    ax.annotate('', xy=(post_ham_x - 0.1, post_ham_y + 0.05),
                xytext=(x_ham[-2], y_ham[-2]),
                arrowprops=dict(arrowstyle='->', color=COLORS['ffn_ham'], lw=2))

    # Legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['embedding'],
               markersize=10, label='Prior'),
        Line2D([0], [0], linestyle='--', color=COLORS['ffn_vfe'], lw=2,
               label=r'VFE: $\nabla F$ descent'),
        Line2D([0], [0], linestyle='-', color=COLORS['ffn_ham'], lw=2,
               label=r'Ham: $H$-conserving'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=8, framealpha=0.9)

    # Manifold label
    ax.text(5, 5.5, r'Statistical Manifold $\mathcal{M}$', ha='center', fontsize=10,
            color=COLORS['manifold'], style='italic')


def create_full_figure():
    """Create the complete figure with all panels."""
    fig = plt.figure(figsize=(14, 10))

    # Create grid
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 0.8], hspace=0.3, wspace=0.2)

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
                 fontsize=14, fontweight='bold', y=0.98)

    # Add caption
    caption = (
        "Both architectures use KL-divergence based attention with gauge-equivariant parallel transport.\n"
        "VFE descent minimizes free energy via gradient flow. Hamiltonian dynamics conserves energy via symplectic integration.\n"
        "The self-consistency term KL(q||p) anchors beliefs to embedding priors, enabling gradient flow to learn embeddings."
    )
    fig.text(0.5, 0.02, caption, ha='center', fontsize=9, style='italic',
             wrap=True, color='gray')

    return fig


def main():
    """Generate and save the figure."""
    fig = create_full_figure()

    # Save in multiple formats
    fig.savefig('figures/gauge_transformer_architecture.pdf',
                dpi=300, bbox_inches='tight', pad_inches=0.1)
    fig.savefig('figures/gauge_transformer_architecture.png',
                dpi=300, bbox_inches='tight', pad_inches=0.1)

    print("✓ Saved: figures/gauge_transformer_architecture.pdf")
    print("✓ Saved: figures/gauge_transformer_architecture.png")

    plt.show()


if __name__ == '__main__':
    main()
