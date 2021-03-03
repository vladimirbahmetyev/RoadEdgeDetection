# RoadEdgeDetection
Defining the boundaries of a snow-covered road
## Prerequisites
For properly work you need to have python 3.7+ and to install dependencies from requiriments.txt.
## Installing 
For installing you can use simple pip command.

`pip install -r requirements.txt`

## General overview
This project is developing to solve the problem of recognizing a snow-covered road. The program is waiting for .mp4 files on input and give .mp4 files on output with  colorized clusters now.
Now three stages have been implemented:
1. Detecting pixel by Sobel Operator
2. Calculating Gradient's angle
3. Clusterization pixels by angle

## Other features
The project also includes:
* Creating summary frame by bufferization frames with boundary pixels.
* Optimization by Numba Python compiler.
* Creating frame's stats for analyzing optimal threshold.

## Features, wich will be implemented in future
* UI/API
* Calculation proposed road boundaries.
* Caching road boundaries.
* More optimization for real time.
