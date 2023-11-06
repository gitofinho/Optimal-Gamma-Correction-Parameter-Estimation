# Optimal-Gamma-Correction-Parameter-Estimation

My First Research topic in CILAB(Computational Image laboratory). 

We propose an efficient low-light image enhancement algorithm based on an optimization-based approach for gamma correction parameter estimation. We first separate an input color image into the luminance and chrominance channels, and then normalize the luminance channel using the logarithmic function to make it consistent with the human perception. Then, we divide the luminance image into dark and bright regions, and estimate the optimal gamma correction parameter for each region independently. Specifically, based on the statistical properties of the input image, we formulate a convex optimization problem that maximizes the image contrast subject to the constraint on the gamma value. By efficiently solving the optimization problems using the convex optimization theories, we obtain the optimal gamma parameter for each region. Finally, we obtain an enhanced image by merging the independently enhanced dark and bright regions with the optimal gamma parameters. Experimental results on real-world images demonstrate that the proposed algorithm can provide higher enhancement performance than state-of-the-art algorithms in terms of both subjective and objective evaluations, while providing a substantial improvement in speed.

## Publication

[SCIE] Inho Jeong and Chul Lee, “An optimization-based approach to gamma correction parameter estimation for low-light image enhancement,” Multimedia Tools and Applications, vol. 80, no. 12, pp. 18027–18042, May 2021.

[Best Paper Award] Inho Jeong and Chul Lee, “Low-light video enhancement based on optimal gamma correction parameter estimation,” in Proc. International Workshop on Frontiers of Computer Vision (IW-FCV), Gangneung, Korea, Feb. 2019. (Best Paper Award)
