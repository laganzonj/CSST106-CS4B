# CSST106 - Perception and Computer Vision: Exercise 2 Documentation

## Overview
This document outlines the methodology, observations, and results for each task conducted in **Exercise 2** of the CSST106 course. The exercise involves feature extraction and matching using Python and OpenCV.

## Author
**Jonathan Q. Laganzon**  
**Program**: BSCS-4B

## Tasks and Methodology

### 1. SIFT (Scale-Invariant Feature Transform) Feature Extraction
**Approach**:
- Imported `cv2`, `numpy`, and `matplotlib`.
- Used SIFT to detect key points and compute descriptors.
- Visualized key points using `plt.imshow`.

**Observations**:
- SIFT is accurate but computationally intensive.
- Detects scale-invariant, distinctive features useful for object recognition.

**Results**:
- Key points were visualized on images, showing high feature density in complex areas.

### 2. SURF (Speeded-Up Robust Features) Feature Extraction
**Approach**:
- Similar to SIFT, used SURF with parameter tuning (e.g., `hessianThreshold`).
- Plotted the results using `cv2.drawKeypoints`.

**Observations**:
- SURF is faster than SIFT and suitable for near real-time applications.
- Less detail captured compared to SIFT, but still effective.

**Results**:
- Feature visualization displayed distinctive points with faster computation.

### 3. ORB (Oriented FAST and Rotated BRIEF) Feature Extraction
**Approach**:
- Configured `cv2.ORB_create` to set the number of features.
- Visualized results using `cv2.drawKeypoints`.

**Observations**:
- ORB is efficient for real-time scenarios with acceptable precision.
- Handles most image types well, but less complex than SIFT/SURF.

**Results**:
- Displayed key points highlighting edges and corners.

### 4. Feature Matching
**Approach**:
- Implemented `cv2.BFMatcher` and `cv2.FlannBasedMatcher`.
- Used cross-checking and ratio tests for more reliable matches.

**Observations**:
- Matching performance varies with the matcher used.
- Ratio tests improved match quality.

**Results**:
- Matched key points between images, demonstrating successful correspondence.

### 5. Applications of Feature Matching
**Approach**:
- Used feature matching for object detection and homography estimation.
- Visualized results with `cv2.drawMatches`.

**Observations**:
- Homography worked well with strong matches.
- Filtering matches is crucial for practical applications.

**Results**:
- Images aligned successfully using homography overlays.

### 6. Combining Feature Extraction Methods
**Approach**:
- Ran SIFT and ORB together for richer feature detection.
- Merged results and visualized combined outputs.

**Observations**:
- Combining methods improved overall feature coverage but increased processing time.
- Useful for scenarios requiring detailed analysis.

**Results**:
- Combined output visualizations showed more comprehensive key points.

## Conclusion
- **SIFT**: Best for high accuracy but slow.
- **SURF**: Balances speed and accuracy.
- **ORB**: Best for real-time processing.
- **Combination**: Provides more detail but takes more time.

## Recommendations
- Use **SIFT/SURF** for accuracy-critical tasks.
- Use **ORB** for real-time applications.
- Combine methods if processing time is not an issue.
