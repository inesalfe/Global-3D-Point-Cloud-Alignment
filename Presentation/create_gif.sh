#!/bin/sh

convert -delay 100 -loop 0 $(ls -1 Figures/fig_*.png | sort -V) anim.gif