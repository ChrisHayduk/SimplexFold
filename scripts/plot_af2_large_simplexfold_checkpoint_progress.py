#!/usr/bin/env python3
"""Compare actual saved research-v2 checkpoints, never unbound history maxima."""

from plot_nanofold_experiment_metrics import main as plot_main


def main(argv=None):
    return plot_main(argv, checkpoint_curve=True)


if __name__ == "__main__":
    main()
