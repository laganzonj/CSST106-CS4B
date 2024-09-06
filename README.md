# CSST106-CS4B: Machine Problem No. 1: Exploring the Role of Computer Vision and Image Processing in AI

### Laganzon, Jonathan Q. - BSCS4B

### Introduction to Computer Vision and Image Processing

• Computer Vision is a field of artificial intelligence (AI) that enables machines to interpret and understand visual information, similar to how humans use their eyes and brains to interpret things. It involves acquiring, processing, analyzing, and understanding images or videos. It has a goal of making decisions or providing output based on that visual data. Computer Vision is crucial for applications like facial recognition, object detection, autonomous vehicles, medical imaging, and augmented reality.

•Image processing is a foundational step in computer vision that involves techniques to enhance, transform, or analyze images. image processing enhances, manipulates, and structures this data in ways that make it suitable for analysis. Without proper image processing, AI systems would struggle to handle complex visual tasks and achieve high levels of accuracy.


### Types of Image Processing Techniques

1. Filtering, this technique involves applying a filter to an image to enhance or suppress specific features. Common filters include blurring to smooth images and sharpening to enhance details. Filtering helps in reducing noise, improving image clarity, and highlighting important features.
   
2. Edge detection identifies the boundaries of objects within an image by finding areas where the image intensity changes sharply. Techniques like the Canny or Sobel edge detectors are used to highlight these boundaries, which is useful for detecting shapes and structures within an image.
   
3. Segmentation divides an image into distinct regions or segments based on pixel values or other features. This process helps in isolating objects or areas of interest within an image, making it easier to analyze and process specific parts of the image separately. Segmentation is commonly used in object recognition and image analysis tasks.


### Case Study Overview: Facial Recognition Systems
#### Application

  Facial recognition systems are widely used for identity verification and security, ranging from unlocking smartphones to surveillance systems in public places. These systems utilize computer vision techniques to identify and verify individuals based on their facial features.


##### Challenges 
 Variability in Lighting and Backgrounds. Ensures that variations in lighting and backgrounds do not affect the recognition accuracy.

 Blur and Noise Introduction: acquired images that are blurred or noisy. Blur can occur due to movement or poor focus, and noise can be introduced by low-quality sensors or environmental factors. These issues can degrade recognition performance and make it harder to accurately identify faces.

### Image Processing Implementation

 Super-Resolution (Upscaling) increases image clarity by enlarging the image size, allowing more details to be seen and improving the recognition of facial features, especially in low-resolution images, using methods like Lanczos interpolation to maintain quality. Deblurring fixes blurry images caused by movement or focus issues, making facial features sharper and more distinct by applying filters such as the Wiener filter. Denoising cleans up images by removing random noise and distortions, ensuring clarity and preserving important details, using techniques like Non-Local Means Denoising. Together, these preprocessing steps enhance the performance of facial recognition systems by improving image quality and feature detection.

### Conclusion

 image processing is crucial in AI because it improves the quality and usability of images, which directly impacts the performance of AI models. Techniques like Super-Resolution, Deblurring, and Denoising enhance image clarity, making it easier for models to accurately detect and recognize features. This preprocessing ensures that AI systems can handle various image quality issues and produce more reliable results. From this activity, I learned that proper image preprocessing is essential for achieving high accuracy and efficiency in AI applications, as it prepares the data in a way that maximizes the model’s ability to learn and make accurate predictions.

### Extension Activity:  Emerging Image Processing With  Deep Learning-Based Analysis.

In recent years, advancements in deep learning have brought significant improvements to image processing techniques. These new methods enable computers to analyze and understand visual data more effectively, opening up new possibilities in areas such as facial recognition, medical imaging, and security systems. One emerging approach in this field is the use of Siamese Neural Networks (SNN), a specialized deep learning architecture designed for tasks that require comparing images. SNNs are particularly effective in determining the similarity between two images, making them valuable in applications like facial recognition, signature verification, and object tracking. 

Siamese Neural Networks (SNN) take two images as input and compare their features to determine how similar they are. This makes them ideal for tasks where it's important to check whether two images represent the same object or person. Their ability to learn detailed feature representations has made SNNs a powerful tool in improving accuracy and reliability in various real-world applications.

### Potential Impact On Future AI Systems 

New image processing techniques, such as deep learning-based image analysis approach, like Siamese Neural Networks (SNNs), are set to have a significant impact of how AI systems work in the future. SNNs are great at comparing two images to see how similar they are, which makes them useful in many areas. In facial recognition, SNNs can help identify people more accurately, even when the lighting is bad or the angle is different. For medical images, like CT scans or MRIs, SNNs can compare images over time to spot small changes, helping doctors catch diseases like cancer earlier. In security, SNNs can track people or objects in videos, even if the footage is blurry or low-quality. This makes SNNs useful for improving safety and monitoring systems. Overall, SNNs will make AI better at processing images with more accuracy in different fields.
