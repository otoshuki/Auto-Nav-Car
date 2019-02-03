# Auto-Nav-Car (Kriti 2018)

The computer vision sub-system for a miniature autonomous car. Project undertaken during Kriti-2018

## Libraries and Dependencies-
* OpenCV3
* Numpy
* Sklearn
* Matplotlib (Optional)
* PySerial

## Working
* ### calibrate()
    * Take a single frame as input
    * Adjust trackbars for cropping and threshold
* ### run(thresh)
    * Take frames from input
    * Crop the image according to thresh
    * Convert image to grayscale and take threshold
    * Sample points along the y-axis
    * Select points along the inner lanes
    * Apply regression along left and right samples
    * Find intersection and its angle wrt center
    * Visualize the predicted lines and angles
* ### transmit(angle)
    * Select different speeds for left and right motors for different angles
    * Transmit data using pySerial to Arduino
* ### kernel(size)
    * Create ellipse kernel with size (size,size)
* ### morph(layers, mask)
    *  Select different layers of mophological transformations for different scenarios
    *  Applies blurring, dilation and erosion to remove noise
*  ### regression(data)
    *  Applies regression on the given data and returns intercept and slope of line
