#!/bin/bash

python batchProcessing_profileOnly.py --input /../paraview_files/ --output /basin_points/

node batchProcessing_profileOnly.js
