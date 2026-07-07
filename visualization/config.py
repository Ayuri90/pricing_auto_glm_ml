import seaborn as sns

# THEME GLOBAL
sns.set_theme(
    style="whitegrid",
    context="notebook",
    palette="deep",
    font="sans-serif",
    rc={
        "figure.figsize": (10, 6),

        "axes.titlesize": 15,
        "axes.labelsize": 12,
        "axes.titleweight": "bold",

        "xtick.labelsize": 10,
        "ytick.labelsize": 10,

        "legend.fontsize": 10,

        "grid.alpha": 0.25,
        "grid.linestyle": "--",

        "figure.dpi": 120,

        "lines.linewidth": 2,

        "savefig.bbox": "tight"
    }
)


# PALETTES
MAIN_PALETTE = sns.color_palette("deep")

SEQUENTIAL_PALETTE = "viridis"

DIVERGING_PALETTE = "coolwarm"

PASTEL_PALETTE = sns.color_palette("pastel")

# COLORS
MEAN_COLOR = "crimson"

MEDIAN_COLOR = "navy"

OUTLIER_COLOR = "darkred"


# FIGURES
FIGSIZE = (12, 6)

TITLE_SIZE = 16
LABEL_SIZE = 13
TICK_SIZE = 11
LEGEND_SIZE = 11

GRID_ALPHA = 0.30

# SEABORN STYLE
STYLE = "whitegrid"

PALETTE = "deep"

# COLORS
COLOR_PRIMARY = "#1F77B4"
COLOR_DANGER  = "#D62728"
COLOR_SUCCESS = "#2CA02C"
COLOR_WARNING = "#FF7F0E"
COLOR_INFO    = "#17BECF"
COLOR_NEUTRAL = "#7F7F7F"
