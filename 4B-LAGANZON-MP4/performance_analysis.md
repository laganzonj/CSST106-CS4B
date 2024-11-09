# CSST106 - Perception and Computer Vision: Machine Problem 4


### Jonathan Q. Laganzon
### BSCS-4B



# Task 1: Harris Corner Detection

### Purpose
Harris Corner Detection is a technique to detect corners in an image. Corners are often significant features, as they represent areas of abrupt change in intensity.

```python
def harris_corner_detection(image_path):
    #Load the image in grayscale.
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    #Convert to float32 for processing.
    img_float = np.float32(img)
    #Apply Harris Corner Detection.
    harris_corners = cv2.cornerHarris(img_float, blockSize=2, ksize=3, k=0.04)
    #Dilate the corner points to enhance them.
    harris_corners = cv2.dilate(harris_corners, None)
    #Create a color version of the grayscale image to display the red corner points.
    img_with_corners = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    #Mark corners with red color.
    img_with_corners[harris_corners > 0.01 * harris_corners.max()] = [255, 0, 0]
    #Display the original and corner-detected images side by side.
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Original Image')

    plt.subplot(1, 2, 2)
    plt.imshow(img_with_corners)
    plt.title('Harris Corner Detection')
    plt.show()

#Load the image.
image_path = '/content/drive/MyDrive/2x2/photo_2024-10-26_20-54-56.jpg'
harris_corner_detection(image_path)
```

### Explanation
- **Grayscale Conversion**: The image is converted to grayscale, as corner detection typically requires a single channel.
- **Corner Detection**: The Harris detector is applied to find corner points.
- **Visualization**: Red markers highlight detected corners, providing a visual reference.



# Task 2: HOG (Histogram of Oriented Gradients) Feature Extraction

### Purpose
HOG (Histogram of Oriented Gradients) is used for feature extraction, focusing on capturing object shape and structure based on gradient information.

```python
def hog_feature_extraction(image_path):
    #Load the image in grayscale.
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    #Extract HOG features.
    hog_features, hog_image = hog(img,
                                  orientations=9,
                                  pixels_per_cell=(8, 8),
                                  cells_per_block=(2, 2),
                                  block_norm='L2-Hys',
                                  visualize=True)
    #Adjust the intensity of the HOG image for better visualization.
    hog_image = exposure.rescale_intensity(hog_image, in_range=(0, 10))
    #Display the original and HOG visualized images side by side.
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Original Image')

    plt.subplot(1, 2, 2)
    plt.imshow(hog_image, cmap='gray')
    plt.title('HOG Features Extraction')
    plt.show()

#Load the image.
image_path = '/content/drive/MyDrive/2x2/photo_2024-10-26_14-35-28.jpg'
hog_feature_extraction(image_path)
```

### Explanation
- **HOG Extraction**: Extracts gradient-based features that highlight object contours and edges.
- **Visualization**: The HOG image provides insight into the structure and orientation of features within the image.


# Task 3: ORB Feature Matching

### Purpose
ORB (Oriented FAST and Rotated BRIEF) is an efficient method for feature detection and matching, ideal for real-time applications.

```python
def orb_feature_matching(image_path1, image_path2):
    #Load the two images.
    img1 = cv2.imread(image_path1, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(image_path2, cv2.IMREAD_GRAYSCALE)
    #Resize the images to the same size.
    height, width = img1.shape[:2]
    img2 = cv2.resize(img2, (width, height))
    #Initialize the ORB detector.
    orb = cv2.ORB_create()
    #Detect keypoints and compute descriptors for both images.
    keypoints1, descriptors1 = orb.detectAndCompute(img1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(img2, None)
    #Initialize FLANN-based matcher.
    index_params = dict(algorithm=6, table_number=6, key_size=12, multi_probe_level=1)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    #Match descriptors using FLANN.
    matches = flann.knnMatch(descriptors1, descriptors2, k=2)
    #Filter matches using the Lowe's ratio test.
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)
    #Draw matches.
    img_matches = cv2.drawMatches(img1, keypoints1, img2, keypoints2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    #Display the matching keypoints.
    plt.figure(figsize=(10, 6))
    plt.imshow(img_matches)
    plt.title("ORB Feature Extraction and Matching")
    plt.axis('off')
    plt.show()

#Load the Images.
image_path1 = '/content/drive/MyDrive/2x2/FRONT.jpg'
image_path2 = '/content/drive/MyDrive/2x2/SIDE.jpg'
orb_feature_matching(image_path1, image_path2)
```

### Explanation
- **FLANN-based Matching**: Matches descriptors using approximate nearest neighbors.
- **Lowe’s Ratio Test**: Filters matches based on descriptor distance, keeping only reliable matches.


# Task 4: SIFT and SURF Feature Extraction

### Purpose
SIFT and SURF are advanced feature detectors that are scale- and rotation-invariant, making them robust for object recognition.

```python
def sift_and_surf_feature_extraction(image_path1, image_path2):
    #Load the two images in grayscale.
    img1 = cv2.imread(image_path1, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(image_path2, cv2.IMREAD_GRAYSCALE)
    #Resize the second image to match the size of the first.
    height, width = img1.shape[:2]
    img2 = cv2.resize(img2, (width, height))
    #SIFT Feature Extraction.
    sift = cv2.SIFT_create()
    keypoints1_sift, descriptors1_sift = sift.detectAndCompute(img1, None)
    keypoints2_sift, descriptors2_sift = sift.detectAndCompute(img2, None)
    #SURF Feature Extraction (requires xfeatures2d module).
    surf = cv2.xfeatures2d.SURF_create(hessianThreshold=400)
    keypoints1_surf, descriptors1_surf = surf.detectAndCompute(img1, None)
    keypoints2_surf, descriptors2_surf = surf.detectAndCompute(img2, None)
    #SIFT Keypoints.
    img1_sift = cv2.drawKeypoints(img1, keypoints1_sift, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    img2_sift = cv2.drawKeypoints(img2, keypoints2_sift, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    #SURF Keypoints.
    img1_surf = cv2.drawKeypoints(img1, keypoints1_surf, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    img2_surf = cv2.drawKeypoints(img2, keypoints2_surf, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    #Plot SIFT results.
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img1_sift, cmap='gray')
    plt.title("SIFT Keypoints")
    plt.subplot(1, 2, 2)
    plt.imshow(img2_sift, cmap='gray')
    plt.title("SIFT Keypoints ")
    plt.show()

    #Plot SURF results.
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(img1_surf, cmap='gray')
    plt.title("SURF Keypoints")
    plt.subplot(1, 2, 2)
    plt.imshow(img2_surf, cmap='gray')
    plt.title("SURF Keypoints")
    plt.show()

#Load the Images.
image_path1 = '/content/drive/MyDrive/2x2/gogo.jpg'
image_path2 = '/content/drive/MyDrive/2x2/jojo.jpg'
sift_and_surf_feature_extraction(image_path1, image_path2)
```

### Explanation
- **SIFT and SURF**: Detects keypoints and computes descriptors for use in tasks such as image alignment.
- **Visualization**: Keypoints for SIFT and SURF are displayed to analyze the distribution and density of detected features.


# Task 5: Brute-Force Feature Matching

### Purpose
Brute-Force matching compares each descriptor from one image with every descriptor from another, using Hamming distance for binary descriptors.

```python
def brute_force_feature_matching(image_path1, image_path2):
    #Load the two images in grayscale.
    img1 = cv2.imread(image_path1, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(image_path2, cv2.IMREAD_GRAYSCALE)
    #Resize the second image to match the size of the first.
    height, width = img1.shape[:2]
    img2 = cv2.resize(img2, (width, height))
    #Initialize the ORB detector.
    orb = cv2.ORB_create()
    #Detect keypoints and compute descriptors for both images.
    keypoints1, descriptors1 = orb.detectAndCompute(img1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(img2, None)
    #Initialize Brute-Force Matcher with Hamming distance (suitable for ORB).
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    #Match descriptors.
    matches = bf.match(descriptors1, descriptors2)
    #Sort matches based on distance (best matches first).
    matches = sorted(matches, key=lambda x: x.distance)
    #Draw matches.
    img_matches = cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches[:20], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    #Display the matching keypoints.
    plt.figure(figsize=(10, 6))
    plt.imshow(img_matches)
    plt.title("ORB Matches using Brute-Force")
    plt.axis('off')
    plt.show()

image_path1 = '/content/drive/MyDrive/2x2/gogo.jpg'
image_path2 = '/content/drive/MyDrive/2x2/jojo.jpg'
brute_force_feature_matching(image_path1, image_path2)
```

### Explanation
- **Brute-Force Matching**: Computes matches based on Hamming distance.
- **Match Sorting**: The closest matches are prioritized, showing the best 20 matches.

  
# Task 6: Image Segmentation Using Watershed Algorithm

### Purpose
The Watershed algorithm segments images based on intensity variations, with enhancements to detect specific colors and improve segmentation accuracy.

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

def enhanced_watershed_segmentation(image_path):
    # Step 1: Load the image
    img = cv2.imread(image_path)

    # Step 2: Convert the image to HSV color space
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Step 3: Define broader color ranges to capture all car colors
    lower_white = np.array([0, 0, 200])  
    upper_white = np.array([180, 30, 255])  
    lower_blue = np.array([100, 150, 0]) 
    upper_blue = np.array([140, 255, 255])
    lower_green = np.array([40, 40, 40]) 
    upper_green = np.array([80, 255, 255])  # Up

    # Step 4: Create masks for each color range
    mask_white = cv2.inRange(hsv, lower_white, upper_white)
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
    mask_green = cv2.inRange(hsv, lower_green, upper_green)

    # Combine masks
    combined_mask = cv2.bitwise_or(mask_white, mask_blue)
    combined_mask = cv2.bitwise_or(combined_mask, mask_green)

    # Step 5: Use Gaussian Blur to reduce noise
    combined_mask = cv2.GaussianBlur(combined_mask, (15, 15), 0)  # Increased blur kernel size

    # Step 6: Advanced morphological operations to clean up the mask
    kernel = np.ones((11, 11), np.uint8)  # Larger kernel size for finer details
    closing = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel, iterations=5)  # Increased iterations

    # Step 7: Identify sure background area by dilation
    sure_bg = cv2.dilate(closing, kernel, iterations=5)  # Increased iterations for background

    # Step 8: Identify sure foreground area using adaptive thresholding
    dist_transform = cv2.distanceTransform(closing, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.5 * dist_transform.max(), 255, 0)  # Adjust threshold

    # Step 9: Identify the unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Step 10: Label markers for watershed
    _, markers = cv2.connectedComponents(sure_fg)

    # Add one to all labels so that sure background is not zero, but 1
    markers = markers + 1

    # Mark the unknown region as zero
    markers[unknown == 255] = 0

    # Step 11: Apply the Watershed algorithm
    markers = cv2.watershed(img, markers)

    # Mark boundaries in the original image
    img[markers == -1] = [255, 0, 0]  # Red boundaries

    # Step 12: Find contours and draw them
    contours, _ = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours based on area
    min_area = 500  # Adjusted minimum area threshold
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]

    # Draw filtered contours
    cv2.drawContours(img, filtered_contours, -1, (0, 255, 0), 2)  # Green contours for visual confirmation

    # Step 13: Display the segmented image
    plt.figure(figsize=(10, 6))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("Enhanced Segmented Image with Watershed")
    plt.axis('off')
    plt.show()

# Load the image
image_path = '/content/drive/MyDrive/2x2/gtr.jpg'
enhanced_watershed_segmentation(image_path)

```

### Explanation
- **Color Segmentation**: HSV color masks isolate specific colors for targeted segmentation.
- **Watershed Segmentation**: Applies Watershed to detect object boundaries based on color and distance transformations, enhancing segmentation.


