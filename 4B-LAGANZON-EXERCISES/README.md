## 4B-LAGANZON-EXER1

[4B_LAGANZON_EXER1.ipynb](https://github.com/laganzonj/CSST106-CS4B/blob/e297b51d8f1243abd319a64659cfe9294a6d9f63/4B-LAGANZON-EXERCISES/4B_LAGANZON_EXER1.ipynb)

 #### COLAB NOTEBOOK 
 https://colab.research.google.com/drive/1FRdvBKpk9bznmL_P1IKadDGVhqKkUfrc#scrollTo=5foC0G0XlrQf

### Blurring Techniques and Their Uses:
1. Gaussian Blur: Smooths the image by averaging surrounding pixels, reducing noise and softening details. Commonly used for background blur in photography, it provides a gentle blurring effect but doesn't keep edges sharp.

2. Median Blur: Replaces each pixel with the median value of surrounding pixels, effectively removing noise like salt-and-pepper while keeping edges intact. It's useful for cleaning up noisy images without losing detail, often used in medical imaging.

3. Bilateral Filter: Blurs the image while preserving edge sharpness, making it great for reducing noise in facial images without blurring features. It's ideal for smoothing skin in portrait photography.

4. Motion Blur: Simulates movement by creating a streaking effect. It's used in photography to emphasize motion, like in action shots.

5. Box Filter: Averages pixels in a box-shaped area, providing simple noise reduction and blurring. However, it lacks edge preservation and can make images look unnatural.

6. Unsharp Mask: Despite its name, it sharpens images by enhancing edges. It's commonly used to make image details stand out, especially in photo editing.


![549bada4-64bf-4f87-807a-cc78a7e90e4e](https://github.com/user-attachments/assets/f8550933-10c2-48b0-abf0-d243cc2513a3)

 - Blurring: Gaussian and box filters are effective for softening images, while motion blur is used to create a dynamic or artistic streaking effect. Bilateral and median blurs provide smoother results while preserving important details.
- Noise Reduction: Median blur excels at removing noise, particularly salt-and-pepper noise, while Gaussian blur and the bilateral filter also reduce overall noise effectively.
- Edge Preservation: The bilateral filter is unique in its ability to smoothen images while preserving edges, unlike Gaussian or box filters, which blur edges as well. Median blur also maintains edges while reducing noise.
- Sharpening: Unsharp mask is the primary filter for sharpening, enhancing image details by increasing local contrast.
- Artistic Effect: Motion blur adds artistic motion effects, while Gaussian and box filters are used to create soft, dreamy aesthetics.
- Facial Enhancement: The bilateral filter is ideal for facial enhancement, smoothing skin tones while keeping key features sharp. Gaussian blur can also soften skin textures, but at the expense of some detail loss. Unsharp mask is used to bring out fine details, like eyes or hair, in portraits.
