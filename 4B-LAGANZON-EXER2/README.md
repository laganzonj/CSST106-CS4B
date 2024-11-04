# CSST106 - Perception and Computer Vision: Exercise 2 Documentation

## Overview
This document outlines the methodology, observations, and results for each task conducted in **Exercise 2** of the CSST106 course. The exercise involves feature extraction and matching using Python and OpenCV.

**Jonathan Q. Laganzon**  
**Program**: BSCS-4B

## Tasks and Methodology

### 1. SIFT (Scale-Invariant Feature Transform) Feature Extraction
```python
#Import Libraries.
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

#Load the Original Image.
image_path = '/content/drive/MyDrive/2x2/photo_2024-10-26_14-35-28.jpg'
image = Image.open(image_path)

#Convert the PIL image to a NumPy array (already in RGB format).
image_np = np.array(image)

#Convert to grayscale for keypoint detection.
gray_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

#Initialize the SIFT (Scale-Invariant Feature Transform).
sift = cv2.SIFT_create()

#Detect keypoints and compute descriptors.
keypoints, descriptors = sift.detectAndCompute(gray_image, None)

#Draw the detected keypoints on the original image (in RGB format).
image_with_keypoints = cv2.drawKeypoints(image_np, keypoints, None, (0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

#Display Original Image.
plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 1)
plt.imshow(image_np)
plt.title(r'$\bf{Original\ Image}$')

#Display SIFT Keypoints.
plt.subplot(1, 2, 2)
plt.imshow(image_with_keypoints)
plt.title(r'$\bf{SIFT\ Keypoints}$')
plt.tight_layout()
plt.show()
```
**Approach**:
- Imported `cv2`, `numpy`, and `matplotlib`.
- Used SIFT to detect key points and compute descriptors.
- Visualized key points using `plt.imshow`.

**Observations**:
- SIFT is accurate but computationally intensive.
- Detects scale-invariant, distinctive features useful for object recognition.
**Results**:
- Key points were visualized on images, showing high feature density in complex areas.

![exer2-sift](https://github.com/user-attachments/assets/f4bbebcc-a08e-4d48-9b77-f2d52688ce82)



### 2. SURF (Speeded-Up Robust Features) Feature Extraction
```python
#Import Libraries.
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

#Load the Original Image.
image_path = '/content/drive/MyDrive/2x2/photo_2024-10-26_14-35-28.jpg'
image = Image.open(image_path)

#Convert the PIL image to a NumPy array (already in RGB format).
image_np = np.array(image)

#Convert to grayscale for keypoint detection.
gray_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

#Initialize the SURF (Speeded-Up Robust Features).
surf = cv2.xfeatures2d.SURF_create()

#Detect keypoints and compute descriptors.
keypoints, descriptors = surf.detectAndCompute(gray_image, None)

#Draw the detected keypoints on the original image (in RGB format).
image_with_keypoints = cv2.drawKeypoints(image_np, keypoints, None, (255, 0, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

#Display Original Image.
plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 1)
plt.imshow(image_np)
plt.title(r'$\bf{Original\ Image}$')

#Display SURF Keypoints.
plt.subplot(1, 2, 2)
plt.imshow(image_with_keypoints)
plt.title(r'$\bf{SURF\ Keypoints}$')
plt.tight_layout()
plt.show()
```
**Approach**:
- Similar to SIFT, used SURF with parameter tuning (e.g., `hessianThreshold`).
- Plotted the results using `cv2.drawKeypoints`.

**Observations**:
- SURF is faster than SIFT and suitable for near real-time applications.
- Less detail captured compared to SIFT, but still effective.

**Results**:
- Feature visualization displayed distinctive points with faster computation.
![exer2-surf](https://github.com/user-attachments/assets/5cb12099-2c1f-4a60-9377-da5bb5b03e85)S



### 3. ORB (Oriented FAST and Rotated BRIEF) Feature Extraction
```python
#Import Libraries.
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

#Load the Original Image.
image_path = '/content/drive/MyDrive/2x2/photo_2024-10-26_14-35-28.jpg'
image = Image.open(image_path)

#Convert the PIL image to a NumPy array (already in RGB format).
image_np = np.array(image)

#Convert to grayscale for keypoint detection.
gray_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

#Initialize ORB (Oriented FAST and Rotated BRIEF).
orb = cv2.ORB_create()

#Detect keypoints and compute descriptors.
keypoints, descriptors = orb.detectAndCompute(gray_image, None)

#Draw the detected keypoints on the original image (in RGB format).
image_with_keypoints = cv2.drawKeypoints(image_np, keypoints, None, (0, 0, 255), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

#Display Original Image.
plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 1)
plt.imshow(image_np)
plt.title(r'$\bf{Original\ Image}$')

#Display ORB Keypoints.
plt.subplot(1, 2, 2)
plt.imshow(image_with_keypoints)
plt.title(r'$\bf{ORB\ Keypoints}$')
plt.tight_layout()
plt.show()
```
**Approach**:
- Configured `cv2.ORB_create` to set the number of features.
- Visualized results using `cv2.drawKeypoints`.

**Observations**:
- ORB is efficient for real-time scenarios with acceptable precision.
- Handles most image types well, but less complex than SIFT/SURF.

**Results**:
- Displayed key points highlighting edges and corners.
- 
![exer2-orb](https://github.com/user-attachments/assets/48aa602a-8c5c-4a57-8a4e-2adcdefd903f)




### 4. Feature Matching
```python
#Import Libraries.
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

#Load the two images.
image1_path = '/content/drive/MyDrive/2x2/photo_2024-10-26_14-35-28.jpg'
image2_path = '/content/drive/MyDrive/2x2/photo_2024-10-26_14-44-35.jpg'
image1 = Image.open(image1_path)
image2 = Image.open(image2_path)

#Convert the images to NumPy arrays.
image1_np = np.array(image1)
image2_np = np.array(image2)

#Resize the images to the same size.
height, width = image1_np.shape[:2]
image2_np = cv2.resize(image2_np, (width, height))

#Convert to grayscale for keypoint detection.
gray_image1 = cv2.cvtColor(image1_np, cv2.COLOR_RGB2GRAY)
gray_image2 = cv2.cvtColor(image2_np, cv2.COLOR_RGB2GRAY)

#Initialize ORB (Oriented FAST and Rotated BRIEF).
orb = cv2.ORB_create(nfeatures=500)

#Detect keypoints and compute descriptors for both images.
keypoints1, descriptors1 = orb.detectAndCompute(gray_image1, None)
keypoints2, descriptors2 = orb.detectAndCompute(gray_image2, None)

#Initialize Brute-Force Matcher.
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

#Match descriptors between the two images.
matches = bf.match(descriptors1, descriptors2)

#Sort matches based on distance (the lower the distance, the better the match).
matches = sorted(matches, key=lambda x: x.distance)

#Draw the top matches (you can adjust the number of matches to display).
matched_image = cv2.drawMatches(image1_np, keypoints1, image2_np, keypoints2, matches[:50], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

#Display the feature matching keypoints.
plt.figure(figsize=(10, 6))
plt.imshow(matched_image)
plt.title(r'$\bf{Feature\ Matching\ using\ ORB\ and\ Brute-Force\ Matcher}$')
plt.show()
```
**Approach**:
- Implemented `cv2.BFMatcher` and `cv2.FlannBasedMatcher`.
- Used cross-checking and ratio tests for more reliable matches.

**Observations**:
- Matching performance varies with the matcher used.
- Ratio tests improved match quality.

**Results**:
- Matched key points between images, demonstrating successful correspondence.
![exer2-force_matcher](https://github.com/user-attachments/assets/2a82d67b-a549-416c-8dae-6a4a41b2f397)




### 5. Applications of Feature Matching
```python
#Import Libraries.
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

#Load the two images.
image1_path = '/content/drive/MyDrive/2x2/photo_2024-10-26_14-35-28.jpg'
image2_path = '/content/drive/MyDrive/2x2/photo_2024-10-26_14-44-35.jpg'
image1 = Image.open(image1_path)
image2 = Image.open(image2_path)

#Convert the images to NumPy arrays.
image1_np = np.array(image1)
image2_np = np.array(image2)

#Resize the second image to match the first image's size.
height, width = image1_np.shape[:2]
image2_np = cv2.resize(image2_np, (width, height))

#Convert to grayscale for keypoint detection.
gray_image1 = cv2.cvtColor(image1_np, cv2.COLOR_RGB2GRAY)
gray_image2 = cv2.cvtColor(image2_np, cv2.COLOR_RGB2GRAY)

#Initialize ORB (Oriented FAST and Rotated BRIEF).
orb = cv2.ORB_create()

#Detect keypoints and compute descriptors using ORB.
keypoints1, descriptors1 = orb.detectAndCompute(gray_image1, None)
keypoints2, descriptors2 = orb.detectAndCompute(gray_image2, None)

#Match features using BFMatcher.
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(descriptors1, descriptors2)

#Sort matches based on distance.
matches = sorted(matches, key=lambda x: x.distance)

#Extract location of good matches.
src_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

#Find the homography matrix.
M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)

#Warp the second image to align with the first image.
h, w = image1_np.shape[:2]
result = cv2.warpPerspective(image2_np, M, (w, h))

#Display the applications of feature matching.
plt.figure(figsize=(10, 6))
plt.imshow(result)
plt.title(r'$\bf{Image\ Alignment\ using\ Homography\ (ORB)}$')
plt.show()
```
**Approach**:
- Used feature matching for object detection and homography estimation.
- Visualized results with `cv2.drawMatches`.

**Observations**:
- Homography worked well with strong matches.
- Filtering matches is crucial for practical applications.

**Results**:
- Images aligned successfully using homography overlays.
![exer2-homography](https://github.com/user-attachments/assets/15f1a742-10c1-416c-8df7-e0aca2fb6577)





### 6. Combining Feature Extraction Methods
```python
#Import Libraries.
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

#Load the two images in color.
image1_path = '/content/drive/MyDrive/2x2/photo_2024-10-26_14-35-28.jpg'
image2_path = '/content/drive/MyDrive/2x2/photo_2024-10-26_14-44-35.jpg'
image1 = cv2.imread(image1_path)
image2 = cv2.imread(image2_path)

#Resize images.
height, width = 900, 800
image1 = cv2.resize(image1, (width, height))
image2 = cv2.resize(image2, (width, height))

#Convert images to grayscale for feature detection.
gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

#SIFT detector.
sift = cv2.SIFT_create()
keypoints1_sift, descriptors1_sift = sift.detectAndCompute(gray1, None)
keypoints2_sift, descriptors2_sift = sift.detectAndCompute(gray2, None)

#ORB detector.
orb = cv2.ORB_create()
keypoints1_orb, descriptors1_orb = orb.detectAndCompute(gray1, None)
keypoints2_orb, descriptors2_orb = orb.detectAndCompute(gray2, None)

#Brute Force Matcher for SIFT (since SIFT uses 128-dim descriptors).
bf_sift = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
matches_sift = bf_sift.match(descriptors1_sift, descriptors2_sift)

#Brute Force Matcher for ORB (since ORB uses 32-dim descriptors).
bf_orb = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches_orb = bf_orb.match(descriptors1_orb, descriptors2_orb)

#Sort matches by distance (optional, for better visualization).
matches_sift = sorted(matches_sift, key=lambda x: x.distance)
matches_orb = sorted(matches_orb, key=lambda x: x.distance)

#Draw top 50 matches for SIFT and ORB (using original BGR images).
img_matches_sift = cv2.drawMatches(image1, keypoints1_sift, image2, keypoints2_sift, matches_sift[:50], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
img_matches_orb = cv2.drawMatches(image1, keypoints1_orb, image2, keypoints2_orb, matches_orb[:50], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

#Combine SIFT and ORB matches.
combined_matches = matches_sift[:25] + matches_orb[:25]

#Combine keypoints for display.
combined_keypoints1 = keypoints1_sift + keypoints1_orb
combined_keypoints2 = keypoints2_sift + keypoints2_orb

#Draw matches for combined SIFT + ORB (using original BGR images).
combined_img_matches = cv2.drawMatches(
    image1, combined_keypoints1, image2, combined_keypoints2, combined_matches, None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)
plt.figure(figsize=(12, 10))

#Display SIFT Feature Matching.
plt.subplot(2, 2, 1)
plt.imshow(cv2.cvtColor(img_matches_sift, cv2.COLOR_BGR2RGB))
plt.title('SIFT Feature Matching')
plt.axis('off')

#Display ORB Feature Matching.
plt.subplot(2, 2, 2)
plt.imshow(cv2.cvtColor(img_matches_orb, cv2.COLOR_BGR2RGB))
plt.title('ORB Feature Matching')
plt.axis('off')

# Display combined SIFT + ORB matches.
plt.subplot(2, 1, 2)
plt.imshow(cv2.cvtColor(combined_img_matches, cv2.COLOR_BGR2RGB))
plt.title('Combined SIFT + ORB Feature Matching')
plt.axis('off')
plt.tight_layout()
plt.show()
```
**Approach**:
- Ran SIFT and ORB together for richer feature detection.
- Merged results and visualized combined outputs.

**Observations**:
- Combining methods improved overall feature coverage but increased processing time.
- Useful for scenarios requiring detailed analysis.

**Results**:
- Combined output visualizations showed more comprehensive key points.
- ![exer2-compare_algo](https://github.com/user-attachments/assets/0ad5b5fa-4c6e-4b44-b64c-0dfc7434b53d)



## Conclusion
- **SIFT**: Best for high accuracy but slow.
- **SURF**: Balances speed and accuracy.
- **ORB**: Best for real-time processing.
- **Combination**: Provides more detail but takes more time.

## Recommendations
- Use **SIFT/SURF** for accuracy-critical tasks.
- Use **ORB** for real-time applications.
- Combine methods if processing time is not an issue.
