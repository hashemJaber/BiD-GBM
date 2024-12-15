
# Bi-Directional Generation Between Images and Point Clouds Using Argoverse 2 Dataset

## Overview
This project explores **bi-directional generation** between LiDAR point clouds and camera images using the **Argoverse 2 Dataset**. The goal is to evaluate whether deep learning models can reconstruct images from LiDAR data and vice versa, enabling seamless data representation and improving autonomous driving capabilities.

## Table of Contents
1. [Motivation](#motivation)
2. [Dataset](#dataset)
3. [Approach](#approach)
4. [Architecture](#architecture)
5. [Results](#results)
6. [Challenges and Future Work](#challenges-and-future-work)
7. [Dependencies](#dependencies)
8. [How to Run](#how-to-run)
9. [Visuals](#visuals)
10. [References](#references)

---

## Motivation
1. **Transfer Learning for Robotics:** Applications in autonomous systems, such as Tesla's self-driving platforms.
2. **Fault Recovery Systems:** If a LiDAR sensor fails (e.g., pitch black environments), cameras can substitute, and vice versa.
3. **Depth Estimation Applications:** Enhancing environmental mapping for autonomous navigation.

---

## Dataset
We utilized the **Argoverse 2 Dataset**, which provides:
- **7 Ring Cameras** for image collection.
- **2 LiDAR sensors** for spatial mapping.
- Multiple modalities for depth and direction representation.

The dataset contains structured data, including:
- RGB Images (`sensor/camera`)
- LiDAR Point Clouds (`sensor/lidar`)

For reference:
![Argoverse Dataset](https://github.com/user-attachments/assets/127a0ac6-a5da-45b7-b464-0417c92b40c0)

---

## Approach
The project involves two tasks:
1. **Generating Images from LiDAR**:
   - LiDAR input: (x, y, z) coordinates.
   - Preprocessing: Pad data to 10,500 points, discard extraneous dimensions.
   - Model loss: Mean Squared Error (MSE).

2. **Generating LiDAR from Images**:
   - Image input: RGB images.
   - Output: Reconstructed LiDAR (BEV/Bird's Eye View).

The architecture combines **Multi-Task Learning** (MTL) and a **Siamese Neural Network**:
- **MTL** for directionality flags (front, back, sides).
- **Siamese Neural Network** for shared embeddings.

---

## Architecture
1. **Input Data**:
   - LiDAR: `(10500, 3)` → padded and processed.
   - Images: `1550 x 2048 x 3` → RGB channels.

2. **Multi-Task Learning**:
   - Shared feature extraction layers for multiple tasks.
   - Directionality flags provide task-specific learning.

3. **Siamese Network**:
   - Shared weights for embedding extraction.
   - CNN backbone for image/point cloud processing.

Code Snippet:
```python
self.image_conv = nn.Sequential(
    nn.Conv2d(image_channels, 64, kernel_size=3, stride=2, padding=1),
    nn.BatchNorm2d(64),
    nn.ReLU(),
    nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
    nn.BatchNorm2d(128),
    nn.SELU(),
    nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
    nn.BatchNorm2d(256),
    nn.ReLU(),
    nn.Flatten()
)
```

---

## Results
- **Generating Images from LiDAR**:
   Despite using MSE loss, the model struggles to reconstruct street-level details due to a lack of color information in LiDAR data.

- **Generating LiDAR from Images**:
   Output LiDAR data achieves a Bird's Eye View (BEV) but remains imprecise for intricate object mapping.

### Loss Analysis
![Loss Analysis](https://github.com/user-attachments/assets/b47d77c1-6a00-479f-a018-6c8056110641)

### Validation Study
![Validation](https://github.com/user-attachments/assets/4649395f-66a4-4a7f-bf1b-55b525366104)

---

## Challenges and Future Work
- **LiDAR Limitations**: Lacks color and texture information.
- **Scene Reconstruction**: Models fail on intricate features like ground signs.
- **Next Steps**:
   - Remove color data to simplify input processing.
   - Use object detection models like **YOLOv11** for object abstraction.

---

## Dependencies
- **Python** (3.8+)
- **PyTorch** (1.9+)
- **OpenCV**
- **NumPy**
- **Argoverse 2 API**

---

## How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/BiD-GBM.git
   cd BiD-GBM
   ```
---

## Visuals
### Generating Images from LiDAR
![Image Generation](https://github.com/user-attachments/assets/79126dcc-83f9-4552-8ea0-2f90cfa5fcee)

### Generating LiDAR from Images
![LiDAR Reconstruction](https://github.com/user-attachments/assets/b47d77c1-6a00-479f-a018-6c8056110641)

---

## References
- [Argoverse 2 Dataset](https://www.argoverse.org/)
- Multi-Task Learning: **Improved Generalization** through shared layers.
- LiDAR Point Cloud Processing.

---
LICENCE 
TBD
