main.py contains all main functions, including computeH, warpImage, and the mosaicing function, for project 4a.
    - the "mosaic" function executes the entire pipeline, with an "automatic" parameter that can be set to true or false
    - if false, the pipeline from project 4a is executed, by calling the functions within main.py
    - if true, the pipeline for project 4b is executed, by calling "get_best_H" from harris.py
harris.py contains the provided code, along with all functions used in project 4b.
    - get_best_H executes the entire pipeline of automatically computing the best homography matrix
labelling.py is the same script I used for labelling images in Project 3
laplacian_blending.py contains functions from project 2 that are necessary to blend images together.