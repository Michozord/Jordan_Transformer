import matplotlib.pyplot as plt

def matplotlib_update():
    plt.rcParams.update({
        # Font Settings
        'font.family': 'serif',
        'font.serif': ['Times New Roman'],  # Standard for academic papers
        'font.size': 10,
        'mathtext.fontset': 'custom',
        'mathtext.rm': 'serif',          # Roman (regular) math font
        'mathtext.it': 'serif:italic',   # Italic math font
        'mathtext.bf': 'serif:bold',     # Bold math font
        
        # Axes Settings
        'axes.linewidth': 0.8,
        'axes.labelsize': 10,
        'axes.titlesize': 12,
        'axes.spines.top': False,     # Cleaner look by removing top/right spines
        'axes.spines.right': False,
        
        # Tick Settings
        'xtick.direction': 'in',      # Ticks pointing inward is common in journals
        'ytick.direction': 'in',
        'xtick.major.width': 0.6,
        'ytick.major.width': 0.6,
        
        # Line and Marker Settings
        'lines.linewidth': 1.5,
        'lines.markersize': 4,
        
        # Figure and Export Settings
        'figure.figsize': (5, 3.5),    # 5x3.5 inches fits well in a two-column layout
        'figure.dpi': 300,             # High resolution for print
        'savefig.format': 'pdf',       # Vector formats are preferred for publication
        'savefig.bbox': 'tight',       # Prevents labels from being cut off
    })