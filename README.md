# Face Detection with MTCNN

Welcome to the Face Detection repository utilizing the MTCNN (Multi-Task Cascaded Convolutional Networks) model! This project employs MTCNN for detecting faces in both images and videos.

## Repository Structure:

- **InputFolderForImageAndVideoFiles:** This folder contains the input images and videos on which you want to perform face detection.

- **OutputFolder:** The output of the code will be saved in this folder. It includes annotated images/videos with detected faces.

- **FaceRecognition_MTCNN.py:** The main Python script that implements face detection using the MTCNN model.

## Dependencies:

Make sure to install the required dependencies before running the code.

```bash
pip install opencv-python mtcnn matplotlib
```

## Instructions to Run:

1. **Clone the Repository:**

```bash
git clone https://github.com/your-username/face-detection-mtcnn.git
cd face-detection-mtcnn
```

2. **Run the Code:**

```bash
python FaceRecognition_MTCNN.py
```

The script will process the images and videos in the input folder and save the results in the output folder.

## Code Functionality:

The provided Python script, `FaceRecognition_MTCNN.py`, is designed to perform face detection using the MTCNN model. Here's an overview of how the code operates:

### Image Processing:

1. **Image Loading:**
   - The script reads images from the specified `InputFolderForImageAndVideoFiles` directory.

2. **Preprocessing (Optional):**
   - Implement preprocessing steps to enhance image quality or improve face detection accuracy. This may include operations such as resizing, normalization, or noise reduction.

3. **Face Detection:**
   - Utilizing the MTCNN model, faces are detected in each image.

4. **Annotating Detected Faces:**
   - Detected faces are outlined with rectangles, and the processed images are saved in the `OutputFolder` directory.

5. **User Input:**
   - The user is prompted to enter the actual number of faces in the image for later comparison.

6. **Comparison and Logging:**
   - The script compares the detected faces with the user-provided actual count.
   - The results, including the detected faces, actual faces, and differences, are logged.

### Video Processing:

1. **Video Loading:**
   - Similarly, the script reads videos from the specified `InputFolderForImageAndVideoFiles` directory.

2. **Preprocessing (Optional):**
   - Implement preprocessing steps to enhance video frames or improve face detection accuracy. This may include operations such as resizing, normalization, or noise reduction.

3. **Face Detection in Frames:**
   - Frames from the video are processed individually for face detection.

4. **Annotating Detected Faces in Frames:**
   - Detected faces in each frame are outlined, and the processed frames are saved in a new video file.

5. **User Input for Video:**
   - The user is prompted to enter the total actual number of faces in the video.

6. **Comparison and Logging for Video:**
   - The script compares the detected faces in each frame with the user-provided total count.
   - The results, including the detected faces, actual faces, and differences over frames, are logged.

### Plotting Results:

1. **Image Results Plotting:**
   - If image processing occurs, a plot is generated showing the number of detected faces and the actual number of faces for each processed image.

2. **Video Results Plotting:**
   - If video processing occurs, a plot is generated showing the number of detected faces in each frame and the total actual number of faces in the video.

## Scope for Use:

This repository can be used for various applications, including:

1. **Security Systems:** Implement face detection for surveillance systems.
   
2. **Data Annotation:** Automatically annotate faces in images and videos.

3. **Video Analysis:** Analyze and count faces in videos for various purposes.

4. **User Interaction:** Use face detection for applications involving user interaction.

## Additional Expansion:

1. **Frame Skipping:**
   - Adjust the `frame_skip` parameter in the `process_video` function to control the number of frames processed. This can be useful for faster video processing on large datasets.

2. **Integration with Other Models:**
   - Explore the possibility of integrating other face detection models to compare performance and accuracy.

3. **Real-Time Processing:**
   - Modify the code to perform face detection in real-time using a webcam.

4. **Integration with Deep Learning Models:**
   - Investigate opportunities to integrate deep learning models for enhanced face detection accuracy.

## Adding Preprocessing Steps:

1. **Preprocessing for Images:**
   - Implement additional preprocessing steps in the `process_image` function to enhance image quality or improve face detection accuracy. Experiment with operations such as resizing, normalization, or noise reduction.

2. **Preprocessing for Videos:**
   - Extend the `process_video` function to include preprocessing steps for individual frames. This may involve resizing, normalization, or other operations to improve face detection accuracy in videos.

## Documentation Improvement:

Enhance the documentation to provide clear instructions on integrating preprocessing steps for images and videos. Explain the potential benefits of preprocessing for face detection accuracy and usability.

We welcome contributions and suggestions to make this repository more versatile and useful! If you encounter any issues or have ideas for improvement, please open an issue or submit a pull request.

Happy coding! 🚀
