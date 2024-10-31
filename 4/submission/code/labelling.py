import csv
import cv2
import numpy as np
import os

points = []
new_points = []
reset = False


def click_and_label(event, x, y, idk, idc):
    global points, new_points, reset

    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        new_points.append((x, y))
        print(f"Point selected: {x}, {y}")

    elif event == cv2.EVENT_RBUTTONDOWN:
        points = []
        new_points = []
        reset = True
        print("Resetting points...")


def label_points(image_path, out_path, output_frame_path):
    global points, new_points, reset

    image = cv2.imread(image_path)
    clone = image.copy()

    if os.path.exists(out_path):
        with open(out_path, 'r') as file:
            reader = csv.reader(file)
            existing_points = [(int(row[0]), int(row[1])) for row in reader]
            points = existing_points.copy()  # Load existing points
            print(f"Loaded existing points from {out_path}")
    else:
        existing_points = []

    combined_image = image.copy()

    cv2.namedWindow("Image and Points")
    cv2.setMouseCallback("Image and Points", click_and_label)

    while True:
        image_with_points = image.copy()
        rad = int(image.shape[0] / 200)
        for i, (x, y) in enumerate(points):
            cv2.circle(image_with_points, (x, y), rad, (0, 255, 0), -1)
            # Draw point numbers directly on the image
            cv2.putText(image_with_points, str(i + 1), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), rad)

        cv2.imshow("Image and Points", image_with_points)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            print("Quitting...")
            break

        elif key == ord("r"):
            points = existing_points.copy()
            new_points = []
            image = clone.copy()
            print("Image reset, all points cleared.")

    cv2.destroyAllWindows()

    if new_points:
        with open(out_path, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(new_points)
        print(f"Appended {len(new_points)} new points to {out_path}")

    # Save the final image with points and numbers overlaid
    cv2.imwrite(output_frame_path, image_with_points)
    print(f"Image with points and labels saved as {output_frame_path}")

    points = []
    new_points = []
    reset = True



if __name__ == "__main__":
    path = '/Users/meenakshimittal/Desktop/cs180/meenakshi-mittal.github.io/4/Project 4'

    im = ('tennis2')
    img_path = f'{path}/rectification/{im}.JPG'
    out_path = f'{path}/rectification/{im}_labels.csv'
    output_frame_path = f'{path}/rectification/{im}_labels.jpg'

    try:
        label_points(img_path, out_path, output_frame_path)
    except Exception as e:
        print(f"Error with image {im}: {e}")
