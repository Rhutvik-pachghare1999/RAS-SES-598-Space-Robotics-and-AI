Part 1: Photogrammetry Reconstruction and Initial Evaluation

Objective:

  The aim of this stage is to create a 3D reconstruction of the Apollo 17 lunar site using classical photogrammetry techniques. The reconstructed model is later used as input for Gaussian       
  Splatting, which we evaluate using standard metrics (PSNR, SSIM) against the ground truth renderings.

Workflow Overview

    Image Alignment and Sparse Reconstruction

    Dense Point Cloud Generation

    Textured Mesh Reconstruction

    Model Export for Downstream Processing

    Rendering and Evaluation via Gaussian Splatting


🗂 Reconstruction Steps and Visual Results
1.  Generate Tie Points
  ![image](https://github.com/user-attachments/assets/373f5a34-74bd-4980-a894-f2c8786048e8)


3.  Point cloud 
![image](https://github.com/user-attachments/assets/d50512ce-9500-429b-8e63-6bc20e3c91a3)


3. Final Exported Model with texture
![image](https://github.com/user-attachments/assets/46967bde-8bad-4e3b-8c36-c4b9a35e9fd5)

The mesh was exported to .ply format for use in Gaussian Splatting.


Using the generated model, Gaussian Splatting was performed to synthesize novel views of the lunar scene. The renderings were compared to ground truth views to assess quality.

Quantitative Evaluation (PSNR, SSIM)
Each of the rendered images was evaluated using PSNR and SSIM. The results are tabulated below:

| Image # | PSNR ↑ | SSIM ↑ |
|---------|--------|--------|
| 1       | 5.70   | 0.027  |
| 2       | 5.91   | 0.009  |
| 3       | 6.25   | 0.013  |
| 4       | 6.83   | 0.018  |
| 5       | 8.79   | 0.297  |
| 6       | 8.08   | 0.215  |
| 7       | 7.87   | 0.196  |
| 8       | 4.96   | 0.200  |
| 9       | 3.79   | 0.117  |
| 10      | 3.48   | 0.053  |
| 11      | 3.45   | 0.090  |
| 12      | 3.47   | 0.006  |
| 13      | 3.89   | 0.027  |
| 14      | 3.59   | 0.056  |
| 15      | 3.93   | 0.073  |
| **Average** | **5.33** | **0.093** |

**Table 1:** PSNR and SSIM results for 15 rendered images using Gaussian Splatting.

A photogrammetric pipeline was used to reconstruct the lunar landing site in 3D.

The dense point cloud and mesh were successfully generated and textured.

Gaussian Splatting produced renderings with a moderate average PSNR of 5.33 and SSIM of 0.093.

These values highlight areas for improvement in view synthesis, possibly due to pose inaccuracies or sparse data coverage.
