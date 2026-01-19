#!/usr/bin/env python3
"""
Generate Figure 2: Experimental Paradigm Visualization

Creates a publication-quality figure showing the RI and PI experimental designs,
similar to Figure 2 in "Unable to Forget" (Falk et al., 2024).

Shows:
- Left panel: Retroactive Interference (RI) paradigm
- Right panel: Proactive Interference (PI) paradigm

For ICML 2026 submission.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import numpy as np
from pathlib import Path

# Publication style settings
plt.rcParams.update({
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'font.size': 10,
    'font.family': 'sans-serif',
    'axes.labelsize': 11,
    'axes.titlesize': 12,
})


def create_prompt_visualization(output_path: Path):
    """
    Create a two-panel figure showing RI and PI experimental paradigms.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 9))

    # Vibrant professional color scheme
    COLORS = {
        'instruction_bg': '#E3F2FD',      # Light blue
        'instruction_border': '#1976D2',   # Blue
        'initial_fact': '#E8F5E9',         # Light green
        'initial_border': '#2E7D32',       # Dark green
        'update': '#FFF8E1',               # Light amber
        'update_border': '#F57C00',        # Orange
        'final_update': '#FFEBEE',         # Light red
        'final_border': '#C62828',         # Dark red
        'query': '#F3E5F5',                # Light purple
        'query_border': '#7B1FA2',         # Purple
        'answer_ri': '#43A047',            # Green
        'answer_pi': '#E53935',            # Red
        'text_dark': '#212121',
        'text_muted': '#757575',
        'ellipsis': '#BDBDBD',
    }

    # ===============================
    # LEFT PANEL: Retroactive Interference
    # ===============================
    ax1 = axes[0]
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 18)
    ax1.axis('off')

    # Panel title with colored background
    title_box = FancyBboxPatch((0.2, 16.8), 9.6, 1.1,
                                boxstyle="round,pad=0.1,rounding_size=0.3",
                                facecolor='#2E7D32', edgecolor='none')
    ax1.add_patch(title_box)
    ax1.text(5, 17.35, '(a) Retroactive Interference (RI)',
             fontsize=13, fontweight='bold', ha='center', va='center', color='white')
    ax1.text(5, 16.95, 'Query: Recall INITIAL value',
             fontsize=10, ha='center', va='center', color='#C8E6C9')

    y_pos = 16.0

    # Instruction box - made taller to fit text
    instruction_box = FancyBboxPatch((0.3, y_pos - 1.4), 9.4, 1.4,
                                     boxstyle="round,pad=0.1,rounding_size=0.2",
                                     facecolor=COLORS['instruction_bg'],
                                     edgecolor=COLORS['instruction_border'], linewidth=2)
    ax1.add_patch(instruction_box)
    ax1.text(5, y_pos - 0.35, 'INSTRUCTION', fontsize=9, fontweight='bold',
             ha='center', va='center', color=COLORS['instruction_border'])
    ax1.text(5, y_pos - 0.75, '"Learn initial facts, then see updates.', fontsize=8,
             ha='center', va='center', style='italic', color=COLORS['text_dark'])
    ax1.text(5, y_pos - 1.05, 'Later, recall the INITIAL values."', fontsize=8,
             ha='center', va='center', style='italic', color=COLORS['text_dark'])

    y_pos -= 2.0

    # Initial Learning section header
    ax1.text(0.5, y_pos, 'INITIAL LEARNING', fontsize=9, fontweight='bold',
             ha='left', va='center', color=COLORS['initial_border'])
    y_pos -= 0.6

    # Initial facts (target for RI - highlighted)
    initial_facts = [
        ('visual art', 'Artist123'),
        ('tools', 'Tool456'),
        ('gemstone', 'Gem789'),
    ]

    # Store position of first initial fact for arrow
    first_fact_y = y_pos - 0.35

    for i, (cat, val) in enumerate(initial_facts):
        box = FancyBboxPatch((0.3, y_pos - 0.65), 9.4, 0.65,
                             boxstyle="round,pad=0.05,rounding_size=0.1",
                             facecolor=COLORS['initial_fact'],
                             edgecolor=COLORS['initial_border'], linewidth=2)
        ax1.add_patch(box)
        ax1.text(5, y_pos - 0.33, f'{cat}: {val}', fontsize=9,
                 ha='center', va='center', family='monospace',
                 fontweight='bold', color=COLORS['initial_border'])
        y_pos -= 0.75

    ax1.text(5, y_pos - 0.15, '... (46 categories total)', fontsize=8,
             ha='center', va='center', style='italic', color=COLORS['ellipsis'])
    y_pos -= 0.7

    # Target indicator - pointing to FIRST fact (visual art: Artist123)
    ax1.annotate('', xy=(9.7, first_fact_y), xytext=(10.8, first_fact_y + 0.3),
                 arrowprops=dict(arrowstyle='->', color=COLORS['answer_ri'], lw=2.5,
                                connectionstyle="arc3,rad=-0.1"))
    ax1.text(11.2, first_fact_y + 0.5, 'TARGET', fontsize=8, fontweight='bold',
             ha='left', va='center', color=COLORS['answer_ri'])
    ax1.text(11.2, first_fact_y + 0.15, '(recall)', fontsize=6,
             ha='left', va='center', color=COLORS['answer_ri'])

    # Interference section
    ax1.text(0.5, y_pos, 'INTERFERENCE (N updates)', fontsize=9, fontweight='bold',
             ha='left', va='center', color=COLORS['update_border'])
    y_pos -= 0.5
    ax1.text(0.5, y_pos, '"The following facts have been updated:"', fontsize=7,
             ha='left', va='center', style='italic', color=COLORS['text_muted'])
    y_pos -= 0.6

    # Update sequence (randomized interleaving)
    updates = [
        ('gemstone', 'Gem111'),
        ('visual art', 'Artist222'),
        ('tools', 'Tool333'),
        ('visual art', 'Artist444'),
    ]

    for i, (cat, val) in enumerate(updates):
        box = FancyBboxPatch((0.3, y_pos - 0.55), 9.4, 0.55,
                             boxstyle="round,pad=0.05,rounding_size=0.1",
                             facecolor=COLORS['update'],
                             edgecolor=COLORS['update_border'], linewidth=1)
        ax1.add_patch(box)
        ax1.text(5, y_pos - 0.28, f'{cat}: {val}', fontsize=8,
                 ha='center', va='center', family='monospace', color=COLORS['update_border'])
        y_pos -= 0.6

    ax1.text(5, y_pos - 0.1, '... (N × 46 updates, randomized)', fontsize=7,
             ha='center', va='center', style='italic', color=COLORS['ellipsis'])
    y_pos -= 0.8

    # Query section
    ax1.text(0.5, y_pos, 'QUERY', fontsize=9, fontweight='bold',
             ha='left', va='center', color=COLORS['query_border'])
    y_pos -= 0.6

    query_box = FancyBboxPatch((0.3, y_pos - 0.65), 9.4, 0.65,
                               boxstyle="round,pad=0.05,rounding_size=0.1",
                               facecolor=COLORS['query'],
                               edgecolor=COLORS['query_border'], linewidth=2)
    ax1.add_patch(query_box)
    ax1.text(5, y_pos - 0.33, '"What was the INITIAL value of visual art?"',
             fontsize=8, ha='center', va='center', family='monospace', fontweight='bold',
             color=COLORS['query_border'])
    y_pos -= 1.0

    # Answer section
    ax1.text(0.5, y_pos, 'CORRECT ANSWER', fontsize=9, fontweight='bold',
             ha='left', va='center', color=COLORS['answer_ri'])
    y_pos -= 0.6

    answer_box = FancyBboxPatch((0.3, y_pos - 0.7), 9.4, 0.7,
                                boxstyle="round,pad=0.05,rounding_size=0.1",
                                facecolor='#C8E6C9',
                                edgecolor=COLORS['answer_ri'], linewidth=2.5)
    ax1.add_patch(answer_box)
    ax1.text(5, y_pos - 0.35, '✓  Artist123  (the FIRST value)',
             fontsize=10, ha='center', va='center', family='monospace',
             fontweight='bold', color=COLORS['answer_ri'])

    # ===============================
    # RIGHT PANEL: Proactive Interference
    # ===============================
    ax2 = axes[1]
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 18)
    ax2.axis('off')

    # Panel title with colored background
    title_box = FancyBboxPatch((0.2, 16.8), 9.6, 1.1,
                                boxstyle="round,pad=0.1,rounding_size=0.3",
                                facecolor='#C62828', edgecolor='none')
    ax2.add_patch(title_box)
    ax2.text(5, 17.35, '(b) Proactive Interference (PI)',
             fontsize=13, fontweight='bold', ha='center', va='center', color='white')
    ax2.text(5, 16.95, 'Query: Recall LATEST value',
             fontsize=10, ha='center', va='center', color='#FFCDD2')

    y_pos = 16.0

    # Instruction box
    instruction_box = FancyBboxPatch((0.3, y_pos - 1.4), 9.4, 1.4,
                                     boxstyle="round,pad=0.1,rounding_size=0.2",
                                     facecolor=COLORS['instruction_bg'],
                                     edgecolor=COLORS['instruction_border'], linewidth=2)
    ax2.add_patch(instruction_box)
    ax2.text(5, y_pos - 0.35, 'INSTRUCTION', fontsize=9, fontweight='bold',
             ha='center', va='center', color=COLORS['instruction_border'])
    ax2.text(5, y_pos - 0.75, '"Learn initial facts, then see updates.', fontsize=8,
             ha='center', va='center', style='italic', color=COLORS['text_dark'])
    ax2.text(5, y_pos - 1.05, 'Later, recall the MOST RECENT values."', fontsize=8,
             ha='center', va='center', style='italic', color=COLORS['text_dark'])

    y_pos -= 2.0

    # Initial Learning section
    ax2.text(0.5, y_pos, 'INITIAL LEARNING', fontsize=9, fontweight='bold',
             ha='left', va='center', color=COLORS['text_muted'])
    y_pos -= 0.6

    # Initial facts (NOT target for PI - muted)
    for i, (cat, val) in enumerate(initial_facts):
        box = FancyBboxPatch((0.3, y_pos - 0.65), 9.4, 0.65,
                             boxstyle="round,pad=0.05,rounding_size=0.1",
                             facecolor='#FAFAFA',
                             edgecolor='#E0E0E0', linewidth=1)
        ax2.add_patch(box)
        ax2.text(5, y_pos - 0.33, f'{cat}: {val}', fontsize=9,
                 ha='center', va='center', family='monospace', color='#9E9E9E')
        y_pos -= 0.75

    ax2.text(5, y_pos - 0.15, '... (46 categories total)', fontsize=8,
             ha='center', va='center', style='italic', color=COLORS['ellipsis'])
    y_pos -= 0.7

    # Interference section
    ax2.text(0.5, y_pos, 'INTERFERENCE (N updates)', fontsize=9, fontweight='bold',
             ha='left', va='center', color=COLORS['update_border'])
    y_pos -= 0.5
    ax2.text(0.5, y_pos, '"The following facts have been updated:"', fontsize=7,
             ha='left', va='center', style='italic', color=COLORS['text_muted'])
    y_pos -= 0.6

    # Update sequence - show only first few, then ellipsis, then final
    updates_pi = [
        ('gemstone', 'Gem111'),
        ('tools', 'Tool333'),
    ]

    for i, (cat, val) in enumerate(updates_pi):
        box = FancyBboxPatch((0.3, y_pos - 0.55), 9.4, 0.55,
                             boxstyle="round,pad=0.05,rounding_size=0.1",
                             facecolor=COLORS['update'],
                             edgecolor=COLORS['update_border'], linewidth=1)
        ax2.add_patch(box)
        ax2.text(5, y_pos - 0.28, f'{cat}: {val}', fontsize=8,
                 ha='center', va='center', family='monospace', color=COLORS['update_border'])
        y_pos -= 0.6

    ax2.text(5, y_pos - 0.1, '... (N × 46 updates, randomized)', fontsize=7,
             ha='center', va='center', style='italic', color=COLORS['ellipsis'])
    y_pos -= 0.7

    # Final update (target for PI - highlighted)
    final_update = ('visual art', 'Artist999')
    final_y = y_pos - 0.35
    box = FancyBboxPatch((0.3, y_pos - 0.7), 9.4, 0.7,
                         boxstyle="round,pad=0.05,rounding_size=0.1",
                         facecolor=COLORS['final_update'],
                         edgecolor=COLORS['final_border'], linewidth=2.5)
    ax2.add_patch(box)
    ax2.text(5, y_pos - 0.35, f'{final_update[0]}: {final_update[1]}', fontsize=9,
             ha='center', va='center', family='monospace',
             fontweight='bold', color=COLORS['final_border'])

    # Target indicator - pointing to FINAL update (visual art: Artist999)
    ax2.annotate('', xy=(9.7, final_y), xytext=(10.8, final_y + 0.3),
                 arrowprops=dict(arrowstyle='->', color=COLORS['answer_pi'], lw=2.5,
                                connectionstyle="arc3,rad=-0.1"))
    ax2.text(11.2, final_y + 0.5, 'TARGET', fontsize=8, fontweight='bold',
             ha='left', va='center', color=COLORS['answer_pi'])
    ax2.text(11.2, final_y + 0.15, '(recall)', fontsize=6,
             ha='left', va='center', color=COLORS['answer_pi'])

    y_pos -= 1.0

    # Query section
    ax2.text(0.5, y_pos, 'QUERY', fontsize=9, fontweight='bold',
             ha='left', va='center', color=COLORS['query_border'])
    y_pos -= 0.6

    query_box = FancyBboxPatch((0.3, y_pos - 0.65), 9.4, 0.65,
                               boxstyle="round,pad=0.05,rounding_size=0.1",
                               facecolor=COLORS['query'],
                               edgecolor=COLORS['query_border'], linewidth=2)
    ax2.add_patch(query_box)
    ax2.text(5, y_pos - 0.33, '"What was the LAST value of visual art?"',
             fontsize=8, ha='center', va='center', family='monospace', fontweight='bold',
             color=COLORS['query_border'])
    y_pos -= 1.0

    # Answer section
    ax2.text(0.5, y_pos, 'CORRECT ANSWER', fontsize=9, fontweight='bold',
             ha='left', va='center', color=COLORS['answer_pi'])
    y_pos -= 0.6

    answer_box = FancyBboxPatch((0.3, y_pos - 0.7), 9.4, 0.7,
                                boxstyle="round,pad=0.05,rounding_size=0.1",
                                facecolor='#FFCDD2',
                                edgecolor=COLORS['answer_pi'], linewidth=2.5)
    ax2.add_patch(answer_box)
    ax2.text(5, y_pos - 0.35, '✓  Artist999  (the LAST value)',
             fontsize=10, ha='center', va='center', family='monospace',
             fontweight='bold', color=COLORS['answer_pi'])

    # Adjust layout
    plt.subplots_adjust(left=0.02, right=0.88, top=0.95, bottom=0.05, wspace=0.15)

    # Save figure
    plt.savefig(output_path, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"✅ Saved: {output_path}")

    # Also save PDF for LaTeX
    pdf_path = output_path.with_suffix('.pdf')
    plt.savefig(pdf_path, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"✅ Saved: {pdf_path}")

    plt.close()


def main():
    """Generate the prompt visualization figure."""
    print("Generating Figure 2: Experimental Paradigm Visualization")
    print("=" * 60)

    # Output paths - run from project root
    import os
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    output_dir = project_root / 'results' / 'key_results'
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / 'figure2_experimental_paradigm.png'

    create_prompt_visualization(output_path)

    print("\n✅ Figure 2 generation complete!")
    print(f"   Output: {output_path}")
    print(f"   PDF:    {output_path.with_suffix('.pdf')}")


if __name__ == '__main__':
    main()
