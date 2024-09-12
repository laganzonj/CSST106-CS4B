# CSST106-CS4B: Machine Problem No. 2: Applying Image Processing Techniques

### Laganzon, Jonathan Q. - BSCS4B
[https://github.com/user-attachments/assets/0b5760a7-7253-45aa-a5cc-7de3bffa709a](https://github.com/user-attachments/assets/b7a696e5-a9db-4411-85c9-e2c9b7d21db3)

### Image Processing Techniques

- Scaling
- Rotation
- Blurring
- Edge Detection

### Main Libraries Used
  #### cv2 (OpenCV):
  cv2 is a Python module that lets you use OpenCV, a powerful tool for working with images and videos. It helps you do tasks like:

  - Reading, writing, and displaying images/videos (e.g., cv2.imread(), cv2.imwrite(), cv2.imshow())
  - Image transformations (e.g., resizing, rotation, scaling)
  - Filters (e.g., blurring, edge detection)
  - Object detection (e.g., face detection)
  - Feature extraction (e.g., corner detection, template matching)

  #### numpy (imported as np):
  numpy is a Python library that helps you work with large sets of numbers arranged in grids (like tables or images). It allows you to:

  - Images are often represented as arrays of pixel values (e.g., 2D arrays for grayscale images, 3D arrays for color images).
  - Numpy allows for efficient manipulation of these arrays, including transformations, filtering, and mathematical operations.


### SCALING
```python
def scale_image(image, scale_factor):
  height, width = image.shape[:2]
  scale_img = cv2.resize(image, (int(width*scale_factor), 
                                 int(height*scale_factor)), 
                                 interpolation=cv2.INTER_LINEAR)
  return scale_img

scaled_image = scale_image(image,0.5)
display_image(scaled_image, "Scaled Image")
```
![SCALED_IMAGE](https://github.com/user-attachments/assets/1e44a051-11f4-414d-99ea-fe52167b2158)

Scaling an image means resizing it, either making it larger or smaller. In the provided code, the image is scaled based on the given `scale_factor`. Since the `scale_factor` is set to `0.5`, the image is reduced to half its original size.

### ROTATION

```python
def rotate_image(image, angle):
  height, width = image.shape[:2]
  center = (width//2, height//2)
  matrix = cv2.getRotationMatrix2D(center, angle, 1)
  rotated_img = cv2.warpAffine(image, matrix, (width, height))
  return rotated_img

rotated_image = rotate_image(image, 45)
display_image(rotated_image, "Rotated Image")
```
![ROTATED_IMAGE](https://github.com/user-attachments/assets/0f96e998-0e13-464f-b766-3b939ac49162)

Rotating an image means turning it around a central point by a specific angle. In the provided code, the image is rotated by the given angle (in this case, 45 degrees). The center of the image is used as the pivot for the rotation, and cv2.warpAffine() applies the transformation

### BLURRING
```python
gussian_blur = cv2.GaussianBlur(image, (15,15), 0)
display_image(gussian_blur, "Gussain Blur")

median_blur = cv2.medianBlur(image, 15)
display_image(median_blur, "Median Blur")

bilateral_blur = cv2.bilateralFilter(image, 51, 15, 15)
display_image(bilateral_blur, "Bilateral Blur")
```
![GAUSSIAN_BLUR](https://github.com/user-attachments/assets/53a0e02f-1509-4d76-ad41-96ee4179e9f2)

![MEDIAN_BLUR](https://github.com/user-attachments/assets/db4f1ef0-8fc2-4e8a-b4d4-59c82140c437)

![BILATERAL_BLUR](https://github.com/user-attachments/assets/d1d15f8a-ae14-45ef-91a6-7be0f6442ee6)
1. Gaussian Blur smooths the image by averaging pixel values, using a (15,15) kernel to reduce noise and detail.
2. Median Blur replaces each pixel with the median of nearby pixels, effectively removing noise while keeping edges sharp with a kernel size of 15.
3. Bilateral Blur smooths the image while preserving edges by considering both pixel distance and color, using a filter size of 51 and intensity values of 15.
   
Each method reduces noise differently, with varying effects on edges.

### EDGE DETECTION
```python
edge = cv2.Canny(image, 200, 50)
display_image(edge, "Edge Detection")
```
![EDGE_DETECTION](https://github.com/user-attachments/assets/51f8b066-5fcc-4903-9f49-ab46582cee89)

Edge Detection using the Canny method identifies the edges in an image by detecting rapid changes in pixel intensity. In the provided code, the values 200 and 50 are thresholds that control how strong an edge must be to be detected. The result highlights the edges while ignoring less distinct areas.

### SUMMARY 

Techniques like scaling, rotating, blurring, and edge detection demonstrate various ways to manipulate and analyze images. Each method serves a specific purpose: scaling adjusts image size, rotating changes orientation, blurring reduces noise, and edge detection identifies boundaries. These techniques help enhance image quality, emphasize key features, and prepare images for further processing or analysis.
