#!/bin/sh

## step 1: evaluation and save results
# output: results/evaluation

# line evaluation
python evaluation/Eval_line_global.py
python evaluation/Eval_line_local.py

# color evaluation
python evaluation/Eval_color_illusion_vs_control.py


## step 2: perform statistical test and visualization
# output: results/plots

# Fig 3
python visualization/make_figure_line_evaluation.py

# Fig 4
python visualization/make_figure_color_evaluation.py
