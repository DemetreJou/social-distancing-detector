# Setup

A script has been setup to simplify the setup and program usage. 

## Extra Instruction for NVIDIA GPU Usage (Advanced) 
Follow [this](https://robocademy.com/2020/05/01/a-gentle-introduction-to-yolo-v4-for-object-detection-in-ubuntu-20-04/) guide to install and setup all CUDA dependencies  
(be warned setting cuda on wsl isn't that straight forward)

## For CPU (simple) and GPU Usage
Download the model weights [here](https://drive.google.com/file/d/1cewMfusmPjYWbrnuJRuKhPMwRe_b9PaT/view) and put the file in the main folder.  
Next, Run the bootstrap script the install all other python depdencies  

Run  `source run bootstrap`

# How to Run the program
Start by running `source run help` to show a list of commands on how to run the different usage options

### Running on the pre-recorded video 
Run the program for fast testing on the pre-recorded shopping mall video  
This will save a video of the detection results to disk

Run  `source run video` 

### Live Detection Mode
Run the live detection and create heatmaps 
First, setup your camera in the correct position 

Run  `source run live [IP]` and provide the IP address of your camera (We used Droid Cam app on android for testing)

This will take a frame from the camera and open the homography picker. Select 4 points in the order provided corredesponding to the floor/corner points. Once complete, the next video will open showing a live social distancing detection and the 2D overhead view. Every minute a copy of the heatmap will be saved to disk. 




