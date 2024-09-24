import cv2
import numpy as np

# Initialize variables
point1 = None  # First corner of the rectangle
point2 = None  # Second corner of the rectangle
clicks = 0  # Track number of clicks (even for point1, odd for point2)


# Mouse callback function to handle the two-point rectangle drawing
def select_points(event, x, y, flags, param):
    global point1, point2, clicks

    if event == cv2.EVENT_LBUTTONDOWN:
        clicks += 1

        if clicks % 2 == 1:
            # First point (corner of the rectangle)
            point1 = (x, y)
        elif clicks % 2 == 0:
            # Second point (opposite corner of the rectangle)
            point2 = (x, y)
            cv2.rectangle(mask, point1, point2, (255, 255, 255), -1)  # Draw the rectangle on the mask


# Load the input image
input_image = cv2.imread('spline/aligned_wiktor.jpg')
height, width = input_image.shape[:2]

# Create a fully black image (mask)
mask = np.zeros((height, width, 3), np.uint8)

# Create a window and bind the mouse callback function
cv2.namedWindow('image')
cv2.setMouseCallback('image', select_points)

while True:
    # Display the input image and mask side by side
    combined_image = np.hstack((input_image, mask))
    cv2.imshow('image', combined_image)

    # Press 'q' to quit or 'r' to reset the mask
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        mask[:] = 0  # Reset mask to fully black

# Save the mask as a file
cv2.imwrite('spline/wikshi_mask.png', mask)

# Clean up
cv2.destroyAllWindows()
