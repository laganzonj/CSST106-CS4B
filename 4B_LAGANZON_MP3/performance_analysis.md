# CSST106 - Perception and Computer Vision: Machine Problem 3: Feature Extraction and Object Detection


### Jonathan Q. Laganzon
### BSCS-4B

# Performance Analysis of Feature Detection and Matching Methods

## Overview

In this notebook, we explore and analyze three feature detection and matching methods commonly used in computer vision: SIFT, SURF, and ORB. Feature detection and matching are fundamental tasks in computer vision, as they allow us to recognize objects, align images, and even reconstruct 3D scenes. This guide will walk through each step to detect features, match keypoints, align images, and analyze the performance of these algorithms.

## Feature Detection Performance Analysis

This analysis compares the number of keypoints detected and the time taken to compute descriptors using SIFT, SURF, and ORB. Each method was applied to two images, and the results are summarized below:

| Method | Keypoints Detected (Image 1) | Keypoints Detected (Image 2) | Time Taken (s) |
|--------|-------------------------------|-------------------------------|-----------------|
| SIFT   | 683                           | 1481                          | 0.480805       |
| SURF   | 1606                          | 1532                          | 1.050636       |
| ORB    | 500                           | 500                           | 0.016875       |

### Discussion
1. **Keypoints Detected**: 
   - **SIFT** detects a moderate number of keypoints in both images, making it a balanced option for applications that require both accuracy and moderate speed.
   - **SURF** detects the most keypoints among the three methods, with 1606 in the first image and 1532 in the second. This high number of keypoints increases robustness but also requires more computational resources.
   - **ORB** detects a fixed number of keypoints (500) in each image, which is considerably lower than SIFT and SURF. However, it is sufficient for many real-time applications where a quick response is required.

2. **Time Taken**: 
   - **SIFT** takes approximately 0.48 seconds to detect and compute descriptors, showing a balance between computation time and the number of features detected.
   - **SURF** is the slowest, taking about 1.05 seconds. Its longer computation time corresponds to the higher number of detected keypoints, making it suitable for applications where more detailed feature extraction is needed and time is not a strict constraint.
   - **ORB** is by far the fastest, with only 0.016 seconds required for feature detection and descriptor computation. This makes ORB ideal for real-time applications, such as mobile and embedded devices, where computational resources are limited.

### Summary
- **SURF** is optimal when a high number of keypoints is necessary, but it is computationally expensive.
- **SIFT** provides a good balance between the number of keypoints and processing time.
- **ORB** is the best choice when speed is a priority, with a trade-off in the number of detected keypoints.

---

## Feature Matching Performance Analysis

In this part, we assess the quality and speed of feature matching using Brute-Force (BF) and FLANN matchers. The results are presented below:

| Method | Matches (Brute-Force) | Time Taken (BF) | Matches (FLANN) | Time Taken (FLANN) |
|--------|------------------------|-----------------|------------------|---------------------|
| SIFT   | 242                    | 0.061040       | 5               | 0.079634           |
| SURF   | 390                    | 0.148497       | 39              | 0.086774           |
| ORB    | 142                    | 0.010855       | 4               | 0.017963           |

### Discussion
1. **Brute-Force Matcher (BF)**:
   - **SIFT** achieves 242 matches with a matching time of around 0.061 seconds. This balance makes it ideal for applications where accuracy is important, though it’s not the fastest option.
   - **SURF** finds the highest number of matches (390) but takes longer (0.148 seconds). The high match count and longer processing time suggest it is best suited for applications requiring detailed image comparison.
   - **ORB** has 142 matches with BF, taking only 0.010 seconds. This makes it the fastest method, although with fewer matches than SIFT and SURF, which may affect accuracy.

2. **FLANN Matcher**:
   - **SIFT** and **SURF** both find significantly fewer matches with FLANN than with BF. This is because FLANN uses approximate matching, which is faster but can miss some correct matches. For SIFT, only 5 matches are found with FLANN in 0.079 seconds, and for SURF, 39 matches are found in 0.086 seconds.
   - **ORB** achieves only 4 matches with FLANN in 0.017 seconds. ORB’s binary descriptors are generally more suitable for BF than FLANN, hence the low match count with FLANN.

3. **Time Comparison**:
   - **BF Matcher**: ORB is the fastest method with BF, followed by SIFT, and then SURF, which is the slowest.
   - **FLANN Matcher**: FLANN is faster for ORB and SURF compared to BF, but it yields significantly fewer matches, especially with SIFT and ORB.

### Summary
- **SIFT** and **SURF** produce a higher number of matches with BF than FLANN, making them more suited for applications requiring high match accuracy.
- **FLANN** is faster but sacrifices the number of matches, making it useful when approximate matches are sufficient, and speed is prioritized.
- **ORB with BF** is the fastest combination, ideal for real-time applications with moderate accuracy needs.

### Final Recommendations
- **SURF with BF** is ideal for tasks where a high number of accurate matches is required and processing time is less critical.
- **SIFT with BF** provides a balanced approach for general-purpose tasks that need both reasonable accuracy and moderate processing speed.
- **ORB with BF** is the best option for applications where speed is the main priority, with a trade-off in accuracy and match count.

This analysis provides insights into the suitability of each method for different applications. Depending on the requirements for speed, accuracy, and computational resources, you can choose the appropriate feature detection and matching technique.



----


# Performance Analysis of Feature Detection and Matching Methods Codes
#### *Overview*

This analysis examines the performance of three feature detection and matching methods—SIFT, SURF, and ORB—using two images: a front view and a side view of the same object. The analysis covers:

- Steps to perform feature detection and matching using SIFT, SURF, and ORB methods on two images
- Feature Extraction: Evaluates the number of keypoints detected and the time taken by each method.
- Feature Matching: Compares the effectiveness of Brute-Force and FLANN matchers for each method, measuring the number of matches and the time taken.

## Steps

### Step 1: Load Images

Loading images is the first step in any image processing pipeline. Here, we import the necessary libraries, load two images from specified paths, and ensure they have the same dimensions to prepare for subsequent feature detection and matching.

```python
#Import Libraries.
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

#Load the two images.
image1_path = '/content/drive/MyDrive/2x2/FRONT.jpg'
image2_path = '/content/drive/MyDrive/2x2/SIDE.jpg'
image1 = Image.open(image1_path)
image2 = Image.open(image2_path)

#Convert the images to NumPy arrays.
image1_np = np.array(image1)
image2_np = np.array(image2)

#Resize the images to the same size.
height, width = image1_np.shape[:2]
image2_np = cv2.resize(image2_np, (width, height))

#Display the two original color images.
plt.figure(figsize=(15, 8))
plt.subplot(1, 2, 1)
plt.title(r'$\bf{Front\ View}$')
plt.imshow(image1_np)
plt.subplot(1, 2, 2)
plt.title(r'$\bf{Side\ View}$')
plt.imshow(image2_np)
plt.show()
```
![mp3-step1](https://github.com/user-attachments/assets/f94e9d8c-166c-412c-9ebb-8be8d8965264)
Discussion
1. Import Libraries: We use cv2 for computer vision tasks, matplotlib for displaying images, numpy for handling numerical operations, and PIL to open images.
2. Load Images: The images are loaded and converted to NumPy arrays, which is a format suitable for OpenCV operations.
3. Resize Images: We resize the second image to match the dimensions of the first, which is essential for consistent processing in later steps.
4. Display: Finally, we visualize the images to verify they were loaded correctly and to give a quick visual reference before feature extraction.





### Step 2: Extract Keypoints and Descriptors Using SIFT, SURF, and ORB

Feature extraction is the process of detecting key points (specific areas of interest) in an image and describing them with unique descriptors. Here, we use three popular feature extraction methods: SIFT, SURF, and ORB.

```python
def extract_features(image1, image2):
    #SIFT.
    sift = cv2.SIFT_create()
    kp1_sift, des1_sift = sift.detectAndCompute(image1_np, None)
    kp2_sift, des2_sift = sift.detectAndCompute(image2_np, None)

    #SURF.
    surf = cv2.xfeatures2d.SURF_create()
    kp1_surf, des1_surf = surf.detectAndCompute(image1_np, None)
    kp2_surf, des2_surf = surf.detectAndCompute(image2_np, None)

    #ORB.
    orb = cv2.ORB_create()
    kp1_orb, des1_orb = orb.detectAndCompute(image1_np, None)
    kp2_orb, des2_orb = orb.detectAndCompute(image2_np, None)

    return (kp1_sift, des1_sift, kp2_sift, des2_sift), (kp1_surf, des1_surf, kp2_surf, des2_surf), (kp1_orb, des1_orb, kp2_orb, des2_orb)

#Extract features.
sift_features, surf_features, orb_features = extract_features(image1_np, image2_np)

#Function to draw keypoints.
def draw_keypoints(image, keypoints):
    return cv2.drawKeypoints(image, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

#Draw keypoints for SIFT.
image1_sift_kp = draw_keypoints(image1_np, sift_features[0])
image2_sift_kp = draw_keypoints(image2_np, sift_features[2])

#Draw keypoints for SURF.
image1_surf_kp = draw_keypoints(image1_np, surf_features[0])
image2_surf_kp = draw_keypoints(image2_np, surf_features[2])

#Draw keypoints for ORB.
image1_orb_kp = draw_keypoints(image1_np, orb_features[0])
image2_orb_kp = draw_keypoints(image2_np, orb_features[2])

#Display the keypoints for each method.
plt.figure(figsize=(15, 12))
plt.subplot(3, 2, 1)
plt.title(r'$\bf{Front\ View\ (SIFT)}$')
plt.imshow(image1_sift_kp)
plt.subplot(3, 2, 2)
plt.title(r'$\bf{Side\ View\ (SIFT)}$')
plt.imshow(image2_sift_kp)
plt.subplot(3, 2, 3)
plt.title(r'$\bf{Front\ View\ (SURF)}$')
plt.imshow(image1_surf_kp)
plt.subplot(3, 2, 4)
plt.title(r'$\bf{Side\ View\ (SURF)}$')
plt.imshow(image2_surf_kp)
plt.subplot(3, 2, 5)
plt.title(r'$\bf{Front\ View\ (ORB)}$')
plt.imshow(image1_orb_kp)
plt.subplot(3, 2, 6)
plt.title(r'$\bf{Side\ View\ (ORB)}$')
plt.imshow(image2_orb_kp)
plt.tight_layout()
plt.show()
```
![mp3-step2](https://github.com/user-attachments/assets/bd124f4f-2239-4bb8-807c-b0ea46d97d0e)
Discussion
1. SIFT (Scale-Invariant Feature Transform): Detects scale- and rotation-invariant features, which makes it useful for robust object recognition.
2. SURF (Speeded-Up Robust Features): Similar to SIFT but optimized for faster execution. It also provides rotation and scale invariance.
3. ORB (Oriented FAST and Rotated BRIEF): Efficient and suitable for real-time applications, ORB provides rotation-invariant features but lacks scale invariance.
4. Keypoint Drawing: We visualize the keypoints detected by each method on both images, allowing us to see how each algorithm selects areas of interest.

The keypoints and descriptors for each method are stored and will be used in the next step for matching.



### Step 3: Feature Matching with Brute-Force and FLANN Matchers

Once we have descriptors from each image, we can match corresponding points between the two images. Here, we use two matching techniques:

- Brute-Force Matcher: Directly compares each descriptor with every other descriptor.
- FLANN (Fast Library for Approximate Nearest Neighbors) Matcher: An optimized matcher that uses approximate nearest neighbors.
```python
#Function for matching descriptors using Brute-Force Matcher.
def match_descriptors_bf(des1, des2):
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    return matches

#Function for matching descriptors using FLANN Matcher.
def match_descriptors_flann(des1, des2):
    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    #Store only the good matches based on the Lowe's ratio test.
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    return good_matches

#Function to draw matches.
def draw_matches(image1, kp1, image2, kp2, matches):
    matched_image = cv2.drawMatches(image1, kp1, image2, kp2, matches, None,
                                     flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    return matched_image

#Perform matching for SIFT.
sift_matches_bf = match_descriptors_bf(sift_features[1], sift_features[3])
sift_matches_flann = match_descriptors_flann(sift_features[1], sift_features[3])

#Perform matching for SURF.
surf_matches_bf = match_descriptors_bf(surf_features[1], surf_features[3])
surf_matches_flann = match_descriptors_flann(surf_features[1], surf_features[3])

#Perform matching for ORB.
orb_matches_bf = match_descriptors_bf(orb_features[1], orb_features[3])
orb_matches_flann = match_descriptors_flann(orb_features[1].astype(np.float32), orb_features[3].astype(np.float32))

#Draw matches for each method.
sift_bf_matches_img = draw_matches(image1_np, sift_features[0], image2_np, sift_features[2], sift_matches_bf)
sift_flann_matches_img = draw_matches(image1_np, sift_features[0], image2_np, sift_features[2], sift_matches_flann)

surf_bf_matches_img = draw_matches(image1_np, surf_features[0], image2_np, surf_features[2], surf_matches_bf)
surf_flann_matches_img = draw_matches(image1_np, surf_features[0], image2_np, surf_features[2], surf_matches_flann)

orb_bf_matches_img = draw_matches(image1_np, orb_features[0], image2_np, orb_features[2], orb_matches_bf)
orb_flann_matches_img = draw_matches(image1_np, orb_features[0], image2_np, orb_features[2], orb_matches_flann)

#Display the matching results.
plt.figure(figsize=(19, 15))
#SIFT using Brute-Force Matcher.
plt.subplot(3, 2, 1)
plt.title(r'$\bf{SIFT\ Matches\ (Brute-Force)}$')
plt.imshow(sift_bf_matches_img)
#SIFT using FLANN Matcher.
plt.subplot(3, 2, 2)
plt.title(r'$\bf{SIFT\ Matches\ (FLANN)}$')
plt.imshow(sift_flann_matches_img)
#SURF using Brute-Force Matcher.
plt.subplot(3, 2, 3)
plt.title(r'$\bf{SURF\ Matches\ (Brute-Force)}$')
plt.imshow(surf_bf_matches_img)
#SURF using FLANN Matcher.
plt.subplot(3, 2, 4)
plt.title(r'$\bf{SURF\ Matches\ (FLANN)}$')
plt.imshow(surf_flann_matches_img)
#ORB using Brute-Force Matcher.
plt.subplot(3, 2, 5)
plt.title(r'$\bf{ORB\ Matches\ (Brute-Force)}$')
plt.imshow(orb_bf_matches_img)
#ORB using FLANN Matcher.
plt.subplot(3, 2, 6)
plt.title(r'$\bf{ORB\ Matches\ (FLANN)}$')
plt.imshow(orb_flann_matches_img)
plt.tight_layout()
plt.show()
```
![mp3-step3](https://github.com/user-attachments/assets/a704a417-6f97-4610-ac98-a8f5d9759abf)
Discussion
1. Brute-Force Matching: Every descriptor in one image is compared to every descriptor in the other. It’s accurate but computationally intensive.
2. FLANN Matching: Uses approximations to speed up matching, which is especially useful for large sets of descriptors. The knnMatch method finds the two nearest neighbors for each descriptor, and we use Lowe’s ratio test to keep only good matches.
3. Visualization: We visualize the matches to assess the quality of alignment between the two images.


### Step 4: Image Alignment Using Homography

With matched keypoints, we can compute a transformation matrix (homography) that aligns one image with the other. This step is useful for image stitching, where we want to overlay one image on top of another.

```python
#Function for matching descriptors using Brute-Force Matcher.
def match_descriptors_bf(des1, des2):
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    return matches

#Function for matching descriptors using FLANN Matcher.
def match_descriptors_flann(des1, des2):
    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    #Store only the good matches based on the Lowe's ratio test.
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    return good_matches

#Function to draw matches.
def draw_matches(image1, kp1, image2, kp2, matches):
    matched_image = cv2.drawMatches(image1, kp1, image2, kp2, matches, None,
                                     flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    return matched_image

#Perform matching for SIFT.
sift_matches_bf = match_descriptors_bf(sift_features[1], sift_features[3])
sift_matches_flann = match_descriptors_flann(sift_features[1], sift_features[3])

#Perform matching for SURF.
surf_matches_bf = match_descriptors_bf(surf_features[1], surf_features[3])
surf_matches_flann = match_descriptors_flann(surf_features[1], surf_features[3])

#Perform matching for ORB.
orb_matches_bf = match_descriptors_bf(orb_features[1], orb_features[3])
orb_matches_flann = match_descriptors_flann(orb_features[1].astype(np.float32), orb_features[3].astype(np.float32))

#Draw matches for each method.
sift_bf_matches_img = draw_matches(image1_np, sift_features[0], image2_np, sift_features[2], sift_matches_bf)
sift_flann_matches_img = draw_matches(image1_np, sift_features[0], image2_np, sift_features[2], sift_matches_flann)

surf_bf_matches_img = draw_matches(image1_np, surf_features[0], image2_np, surf_features[2], surf_matches_bf)
surf_flann_matches_img = draw_matches(image1_np, surf_features[0], image2_np, surf_features[2], surf_matches_flann)

orb_bf_matches_img = draw_matches(image1_np, orb_features[0], image2_np, orb_features[2], orb_matches_bf)
orb_flann_matches_img = draw_matches(image1_np, orb_features[0], image2_np, orb_features[2], orb_matches_flann)

#Display the matching results.
plt.figure(figsize=(19, 15))
#SIFT using Brute-Force Matcher.
plt.subplot(3, 2, 1)
plt.title(r'$\bf{SIFT\ Matches\ (Brute-Force)}$')
plt.imshow(sift_bf_matches_img)
#SIFT using FLANN Matcher.
plt.subplot(3, 2, 2)
plt.title(r'$\bf{SIFT\ Matches\ (FLANN)}$')
plt.imshow(sift_flann_matches_img)
#SURF using Brute-Force Matcher.
plt.subplot(3, 2, 3)
plt.title(r'$\bf{SURF\ Matches\ (Brute-Force)}$')
plt.imshow(surf_bf_matches_img)
#SURF using FLANN Matcher.
plt.subplot(3, 2, 4)
plt.title(r'$\bf{SURF\ Matches\ (FLANN)}$')
plt.imshow(surf_flann_matches_img)
#ORB using Brute-Force Matcher.
plt.subplot(3, 2, 5)
plt.title(r'$\bf{ORB\ Matches\ (Brute-Force)}$')
plt.imshow(orb_bf_matches_img)
#ORB using FLANN Matcher.
plt.subplot(3, 2, 6)
plt.title(r'$\bf{ORB\ Matches\ (FLANN)}$')
plt.imshow(orb_flann_matches_img)
plt.tight_layout()
plt.show()
```
![mp3-step4](https://github.com/user-attachments/assets/8219df8e-7789-4a0a-80da-30c2b7d35fa7)
Discussion
1. Homography Matrix: This matrix represents the transformation needed to map points from one image to another. It’s calculated using RANSAC (Random Sample Consensus) to remove outliers.
2. Image Warping: With the homography matrix, we warp the first image to align with the second. This is especially useful in applications like panoramic image stitching.


### Step 5: Performance Analysis

Measure the performance of each method by analyzing keypoints, matching quality, and time taken.

```python
#Import Libraries.
import cv2
import numpy as np
import time
import pandas as pd

#Function to analyze keypoints and descriptors for a given method.
def analyze_feature_extraction(method, image1, image2):
    start_time = time.time()

    if method == 'SIFT':
        sift = cv2.SIFT_create()
        kp1, des1 = sift.detectAndCompute(image1, None)
        kp2, des2 = sift.detectAndCompute(image2, None)

    elif method == 'SURF':
        surf = cv2.xfeatures2d.SURF_create()
        kp1, des1 = surf.detectAndCompute(image1, None)
        kp2, des2 = surf.detectAndCompute(image2, None)

    elif method == 'ORB':
        orb = cv2.ORB_create()
        kp1, des1 = orb.detectAndCompute(image1, None)
        kp2, des2 = orb.detectAndCompute(image2, None)

    end_time = time.time()

    #Calculate the number of keypoints detected and time taken.
    num_keypoints1 = len(kp1)
    num_keypoints2 = len(kp2)
    time_taken = end_time - start_time

    return num_keypoints1, num_keypoints2, time_taken, des1, des2

#Analyze each feature extraction method.
image1_gray = cv2.cvtColor(image1_np, cv2.COLOR_BGR2GRAY)
image2_gray = cv2.cvtColor(image2_np, cv2.COLOR_BGR2GRAY)

methods = ['SIFT', 'SURF', 'ORB']
results = {method: analyze_feature_extraction(method, image1_gray, image2_gray) for method in methods}

#Create a list for feature detection results.
feature_detection_data = []

for method, data in results.items():
    num_kp1, num_kp2, time_taken, _, _ = data
    feature_detection_data.append({
        "Method": method,
        "Keypoints Detected (Image 1)": num_kp1,
        "Keypoints Detected (Image 2)": num_kp2,
        "Time Taken (s)": time_taken
    })

#Create DataFrame for feature detection results.
feature_detection_df = pd.DataFrame(feature_detection_data)

#Display results for feature detection.
print("Feature Detection Performance Analysis:")
print(feature_detection_df.to_string(index=False))

#Function for matching descriptors using FLANN Matcher.
def match_descriptors_flann(des1, des2):
    des1 = des1.astype(np.float32)
    des2 = des2.astype(np.float32)

    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1, des2, k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    return good_matches

#Function to match descriptors and analyze matching performance.
def analyze_matching(des1, des2):
    start_time_bf = time.time()
    matches_bf = match_descriptors_bf(des1, des2)
    end_time_bf = time.time()

    start_time_flann = time.time()
    matches_flann = match_descriptors_flann(des1, des2)
    end_time_flann = time.time()

    bf_time = end_time_bf - start_time_bf
    flann_time = end_time_flann - start_time_flann

    return len(matches_bf), bf_time, len(matches_flann), flann_time

#Analyze matching performance for each method.
matching_results = {}
for method, data in results.items():
    _, _, _, des1, des2 = data
    des1 = des1.astype(np.float32)
    des2 = des2.astype(np.float32)

    matching_results[method] = analyze_matching(des1, des2)

#Create a list for matching performance results.
matching_performance_data = []

for method, data in matching_results.items():
    num_matches_bf, time_bf, num_matches_flann, time_flann = data
    matching_performance_data.append({
        "Method": method,
        "Matches (Brute-Force)": num_matches_bf,
        "Time Taken (BF)": time_bf,
        "Matches (FLANN)": num_matches_flann,
        "Time Taken (FLANN)": time_flann
    })

#Create DataFrame for matching performance results.
matching_performance_df = pd.DataFrame(matching_performance_data)

#Display results for matching performance.
print("\nFeature Matching Performance Analysis:")
print(matching_performance_df.to_string(index=False))
```
Discussion
1. Performance Metrics: We evaluate each method based on the number of keypoints detected, the number of good matches, and the time taken for both feature detection and matching.
2. Comparison: This step helps us choose the most suitable method based on the computational resources available and the desired level of accuracy.

   
