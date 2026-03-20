



# DVS Final Project
Design of Visual Systems – Spring Term 2026 – Imperial College London

**Team Members:** 
 - Oliver Dai
 - Lynton Sutton
 - India Lloyd-Evans

## Project Summary
We chose the task of number plate reading. We picked this because there was opportunity to apply our learning of multiple detection methods and colour processing. Challenges we faced in our attempts were largely to do with the different colour contrasts and angular positioning of the number plates in different test images from the dataset – isolating the number plate as a rectangular box amidst shadowing also meant crops became too tight. We employed OCR as the ‘reading’ program, which was somewhat accurate, with slight error in recognising similar characters such as ‘7’ and ‘T’, as well as ‘U’ and ‘V’, but this was not something we could solve further. 
Overall, we made many attempts with roughly 17 different methods, of which we have documented 10. Our recommendations are that ‘oliver_v4’, ‘lynton_v1’ and ‘india_v9’ demonstrate the best segmentations of the image processing (the former two), with the latter showing an output for number plate reading. 


## INSTRUCTIONS

### Prerequisites
- You must have Matlab installed
- Required add-ons:
  - Image Processing Toolbox
  - Computer Vision Toolbox
  - Computer Vision Toolbox Model for Text Detection (TODO: maybe phase this one out for a better model)
- Install via Matlab: Home → Add-Ons → Get Add-Ons → search and install the toolboxes above.

### Running the Code
1. Download or clone this project onto your computer.
2. Open the project's `matlab/` folder in Matlab.
3. Open and run `main.m` in Matlab. Be patient as it downloads the dataset on your first run.
4. Select an image to process from the dialog that opens. You may have to navigate inside subfolders to find the images.

### For Contributors
All code and scripts go in the `matlab/` folder.
Non-dataset images and other resources go in the `assets/` folder.
All dataset images go in the `dataset/` folder, which will be **automatically generated** the first time you run the program. This folder is intentionally not saved to Github.

| file | instructions |
|--|--|
| `main.m` | The entrypoint into the program. Edit this file to change the dataset, image loading behavior, etc. |
| `process_image.m` | Contains our image processing logic. The `process_image()` function is the entrypoint. It is the "brains" that determines which image processing workflow to run. |
| `image_processing/*` | Our actual image processing workflows go in here. Make major edits as new files. Try to use functions to split image processing workflows into discrete, logical steps. This is a good organization practice and will help us understand each others' code. |


## Demo
WIP - Demonstrate our code working with screenshots and descriptions.

## Analysis
What Worked Well:

 - Yellow-plate detection using HSV colour masking worked quite well after parameter tuning. Once the correct Hue/Saturation/Value ranges were found, it isolated the plate with minimal noise. 
 - High-sensitivity adaptive binarization (Sensitivity 0.99) combined with morphological operations effectively removed background noise while preserving characters.
 - When experimenting with CRAFT, it reliably located the plate region even under moderate angles and varying lighting.

What Didn't Work: 

 - Whilst Yellow-plate detection using HSV colour masking worked well, it was not robust and only worked for 1 test image. When using different data sets the HSV ranges would need to be tuned. An adaptive version would be necessary.
 - For OCR to correctly read the number plate, the characters of the plate would need to be perfectly isolated, any noise or additional 'blobs' would lead to errors or partial readings.
 - Hough transform was not able to identify vertical lines
 - White front plates proved significantly harder than yellow rear plates because of similarity to car body paint and reflections
 - Purely rule-based methods (HSV + morphology) struggled with strong parallax and diverse lighting conditions

Areas for Improvement: 

 - Full automation of region selection: develop a scoring system that combines CRAFT confidence, aspect ratio, and plate-like shape.
 - Perspective correction: Automatically detect and unwarp trapezoidal plates caused by side angles (currently only basic deskewing is used).
 - Hybrid OCR engine: Integrate EasyOCR or PaddleOCR (via Python) for higher character recognition accuracy on challenging plates.
 - Multi-plate support: Extend the system to handle images containing multiple vehicles.
 - Performance optimisation: Add GPU support for CRAFT and explore newer models such as YOLOv8 for plate detection.




