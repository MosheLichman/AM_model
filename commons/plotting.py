# __author__ = 'dimitrios'
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt


def scatter_plot(x, y, name):
    f, ax = plt.subplots(figsize=(6, 6))

    ax.scatter(x, y, c=".3")
    # ax.set(xlim=(-3, 3), ylim=(-3, 3))

    # Plot your initial diagonal line based on the starting
    # xlims and ylims.
    diag_line, = ax.plot(ax.get_xlim(), ax.get_ylim(), ls="--", c=".3")

    # def on_change(axes):
    #     # When this function is called it checks the current
    #     # values of xlim and ylim and modifies diag_line
    #     # accordingly.
    #     x_lims = ax.get_xlim()
    #     y_lims = ax.get_ylim()
    #     diag_line.set_data(x_lims, y_lims)
    #
    # # Connect two callbacks to your axis instance.
    # # These will call the function "on_change" whenever
    # # xlim or ylim is changed.
    # ax.callbacks.connect('xlim_changed', on_change)
    # ax.callbacks.connect('ylim_changed', on_change)
    plt.savefig(name)
