import os                                                                                                                           # Import the os module for file and folder operations
import cv2                                                                                                                          # Import the OpenCV library for image and video processing
from mtcnn.mtcnn import MTCNN                                                                                                       # Import the MTCNN face detection model
import time                                                                                                                         # Import the time module for time-related operations
import matplotlib.pyplot as plt                                                                                                     # Import the Matplotlib library for plotting

def process_image(image_path, output_folder, detector):
    """
    Process a single image for face detection.

    Args:
        image_path (str): Path to the input image.
        output_folder (str): Path to the output folder for saving processed images.
        detector: Face detection model.

    Returns:
        tuple: Detected faces count, user input count, and success flag.
    """
    detected_faces = []                                                                                                             # List to store the count of detected faces
    print(f"Processing image: {image_path}")                                                                                        # Print the image processing message
    start_time = time.time()                                                                                                        # Record the start time for performance measurement
    
    img = cv2.imread(image_path)                                                                                                    # Read the image using OpenCV
    
    faces = detector.detect_faces(img)                                                                                              # Detect faces in the image
    detected_faces.append(len(faces))
    
    for face in faces:                                                                                                              # Draw rectangles around detected faces
        x, y, w, h = face['box']
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    file_name, file_extension = os.path.splitext(os.path.basename(image_path))                                                      # Save the processed image
    output_path = os.path.join(output_folder, f"{file_name}_detected{file_extension}")
    cv2.imwrite(output_path, img)

    user_input = int(input("Enter the actual number of faces in the image: "))                                                      # Get user input for the actual number of faces
    
    elapsed_time = time.time() - start_time                                                                                         # Calculate elapsed time
    print(f"Output generated: {output_path}")                                                                                       # Print the output path
    print(f"Elapsed time: {elapsed_time:.2f} seconds\n")

    return detected_faces, user_input, True

def process_video(video_path, output_folder, detector, frame_skip=1):
    """
    Process a video for face detection with optional frame skipping.

    Args:
        video_path (str): Path to the input video.
        output_folder (str): Path to the output folder for saving processed video.
        detector: Face detection model.
        frame_skip (int): Number of frames to skip during processing.

    Returns:
        tuple: List of processed frame numbers, detected faces count, success flag.
    """
    frame_count = 0                                                                                                                 # Counter for frames processed
    detected_faces = []                                                                                                             # List to store the count of detected faces
    print(f"Processing video: {video_path}")                                                                                        # Print the video processing message
    start_time = time.time()                                                                                                        # Record the start time for performance measurement
    
    cap = cv2.VideoCapture(video_path)                                                                                              # Open the video file using OpenCV
    file_name, file_extension = os.path.splitext(os.path.basename(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')                                                                                        # Create a video writer for the output
    output_path = os.path.join(output_folder, f"{file_name}_detected{file_extension}")
    out = cv2.VideoWriter(output_path, fourcc, fps, (int(cap.get(3)), int(cap.get(4))))

    while cap.isOpened():
        ret, frame = cap.read()                                                                                                     # Read a frame from the video
        if not ret:
            break
        
        if frame_count % frame_skip == 0:                                                                                           # Process every nth frame (controlled by frame_skip)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            faces = detector.detect_faces(frame)                                                                                    # Detect faces in the frame
            detected_faces.append(len(faces))
            
            for face in faces:                                                                                                      # Draw rectangles around detected faces
                x, y, w, h = face['box']
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            
            out.write(frame)                                                                                                        # Write the frame to the output video

        frame_count += 1
    
    cap.release()                                                                                                                   # Release video capture and writer objects
    out.release()

    elapsed_time = time.time() - start_time                                                                                         # Calculate elapsed time
    print(f"Output generated: {output_path}")                                                                                       # Print the output path
    print(f"Elapsed time: {elapsed_time:.2f} seconds\n")    

    return list(range(1, frame_count + 1, frame_skip)), detected_faces, True

def main():
    detected_faces_img = []                                                                                                         # List to store detected faces count from images
    user_input_img = []                                                                                                             # List to store user input counts from images
    image_processed = False                                                                                                         # Flag indicating whether images are processed
    video_processed = False                                                                                                         # Flag indicating whether videos are processed
       
    folder_path = "D:\Github\FaceRocgnition\FaceDetection_MTCNN\InputFolderForImageAndVideoFIles"                                   # Path to the folder containing images and videos
    
    output_folder = "D:\Github\FaceRocgnition\FaceDetection_MTCNN\OutputFolder"                                                     # Output folder to save processed images and videos
    os.makedirs(output_folder, exist_ok=True)                                                                                       # Create the output folder if it doesn't exist
    
    detector = MTCNN()                                                                                                              # Create a face detector

    for filename in os.listdir(folder_path):                                                                                        # Process images and videos in the folder
        file_path = os.path.join(folder_path, filename)
        if filename.endswith((".jpg", ".jpeg", ".png")):
            
            detected_faces, user_input, image_processed = process_image(file_path, output_folder, detector)                         # Process image and get results
            user_input_img.append(user_input)
            detected_faces_img.extend(detected_faces)
        elif filename.endswith(".mp4"):            
            total_frames, detected_faces_vid, video_processed = process_video(file_path, output_folder, detector, frame_skip=1)    # Process video and get results            
            user_input_vid = int(input("Enter the total actual number of faces in the video: "))                                    # Ask the user for the actual number of faces in the video
    
    if image_processed:                                                                                                             # Plot the number of detected faces and the actual number over frames for images
        plt.plot(detected_faces_img, label='Detected Faces (Images)')
        plt.scatter(range(len(user_input_img)), user_input_img, color='r', label='Actual Faces (Images)')
        plt.xlabel('Image Number')
        plt.ylabel('Number of Faces')
        plt.title('Face Detection in Images')
        plt.legend()        
        
        image_plot_path = os.path.join(output_folder, "detected_faces_plot_images.png")                                             # Save the plot in the output folder
        plt.savefig(image_plot_path)
        plt.show()
    
    if video_processed:                                                                                                             # Plot the number of detected faces and the actual number over frames for videos
        plt.plot(total_frames, detected_faces_vid, label='Detected Faces (Videos)')
        plt.axhline(y=user_input_vid, color='r', linestyle='--', label='Actual Faces (Videos)')
        plt.xlabel('Frame Number')
        plt.ylabel('Number of Faces')
        plt.title('Face Detection in Videos')
        plt.legend()
        
        video_plot_path = os.path.join(output_folder, "detected_faces_plot_videos.png")                                             # Save the plot in the output folder
        plt.savefig(video_plot_path)
        plt.show()

if __name__ == "__main__":
    main()
