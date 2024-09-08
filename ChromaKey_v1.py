import sys
import cv2
import numpy as np

def crop_to_match(image_a, image_b):
    # Get dimensions of image A and image B
    a_height, a_width = image_a.shape[:2]
    b_height, b_width = image_b.shape[:2]

    # Calculate aspect ratios
    aspect_a = a_width / a_height
    aspect_b = b_width / b_height

    # Determine how to crop image B
    if aspect_b > aspect_a:
        # Crop horizontally (width) to match the aspect ratio
        new_width = int(b_height * aspect_a)
        start_x = (b_width - new_width) // 2
        cropped_b = image_b[:, start_x:start_x + new_width]
    else:
        # Crop vertically (height) to match the aspect ratio
        new_height = int(b_width / aspect_a)
        start_y = (b_height - new_height) // 2
        cropped_b = image_b[start_y:start_y + new_height, :]

    # Resize the cropped image to match the dimensions of image A
    resized_b = cv2.resize(cropped_b, (a_width, a_height), interpolation=cv2.INTER_AREA)

    return resized_b

def resize_with_aspect_ratio(image, width=None, height=None, inter=cv2.INTER_AREA):
    # Get the original image dimensions
    (h, w) = image.shape[:2]

    # If both width and height are None, return the original image
    if width is None and height is None:
        return image

    # Calculate the aspect ratio and new dimensions
    if width is None:
        aspect_ratio = height / float(h)
        new_dimensions = (int(w * aspect_ratio), height)
    else:
        aspect_ratio = width / float(w)
        new_dimensions = (width, int(h * aspect_ratio))

    # Resize the image
    resized_image = cv2.resize(image, new_dimensions, interpolation=inter)
    return resized_image

def resize_to_match_width(img1, img2):
    # resize img2 to match the width of img1
    resized_img2 = cv2.resize(img2, (int(img1.shape[1]/2), int(img1.shape[0]/2)))
    return resized_img2

def TaskOne(color_space, image_path):
    print(f'task1: {color_space}, {image_path}')
    # reading the image and resizing it
    image = cv2.imread(image_path)
    # height, width, channels = image.shape
    image = resize_with_aspect_ratio(image, width = (1280//2))
    # image = cv2.resize(image, (int(width/2), int(height/2)))

    # checking for colorspace
    if 'XYZ' in color_space:
        converted_image = cv2.cvtColor(image, cv2.COLOR_BGR2XYZ)
    elif 'Lab' in color_space:
        converted_image = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    elif 'YCrCb' in color_space:
        converted_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    elif 'HSB' in color_space:
        converted_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    else:
        raise ValueError("Unsupported color space")

    # src image to its components
    C1, C2, C3 = cv2.split(converted_image)

    # normalizing the components
    C1 = cv2.normalize(C1, None, 0, 255, cv2.NORM_MINMAX)
    C2 = cv2.normalize(C2, None, 0, 255, cv2.NORM_MINMAX)
    C3 = cv2.normalize(C3, None, 0, 255, cv2.NORM_MINMAX)

    # normalized components to grayscale
    C1 = cv2.cvtColor(C1, cv2.COLOR_GRAY2BGR)
    C2 = cv2.cvtColor(C2, cv2.COLOR_GRAY2BGR)
    C3 = cv2.cvtColor(C3, cv2.COLOR_GRAY2BGR)

    # 2d array view
    top_row = np.hstack((image, C2))
    bottom_row = np.hstack((C1, C3))
    final_image = np.vstack((top_row, bottom_row))

    # display 2d array view
    cv2.imshow('task 1', final_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def extract_person(green_screen_img):
    # Convert image to HSV color space
    hsv = cv2.cvtColor(green_screen_img, cv2.COLOR_BGR2HSV)
    
    # Define range for green color in HSV
    lower_green = np.array([30, 90, 90])
    upper_green = np.array([100, 255, 255])
    
    # Create a mask for green color
    mask = cv2.inRange(hsv, lower_green, upper_green)
    
    # Invert the mask to get the person
    mask_inv = cv2.bitwise_not(mask)
    
    # Extract the person from the image
    person = cv2.bitwise_and(green_screen_img, green_screen_img, mask=mask_inv)
    white_bg = np.ones_like(green_screen_img) * 255
    person_with_white_bg = cv2.add(person, cv2.bitwise_and(white_bg, white_bg, mask=mask))
    
    return person ,person_with_white_bg, mask

def combined_image(path_scenic_img, path_green_screen_img):
    # Load the green screen image and the scenic background image
    green_screen_img = cv2.imread(path_green_screen_img)
    scenic_img = cv2.imread(path_scenic_img)

    # Get the dimensions of both images
    gs_height, gs_width = green_screen_img.shape[:2]
    scenic_height, scenic_width = scenic_img.shape[:2]

    # Calculate padding to align the green screen image at the bottom
    pad_top = scenic_height - gs_height
    pad_bottom = 0
    pad_left = (scenic_width - gs_width) // 2
    pad_right = scenic_width - gs_width - pad_left

    # Add padding to the green screen image to match the size of the scenic image
    padded_gs_img = cv2.copyMakeBorder(
        green_screen_img,
        pad_top, pad_bottom, pad_left, pad_right,
        cv2.BORDER_CONSTANT,
        value=[0, 255, 0]  # Assuming green background padding (or use black [0,0,0] or white [255,255,255])
    )

    # Convert the padded green screen image to HSV for better color detection
    hsv_img = cv2.cvtColor(padded_gs_img, cv2.COLOR_BGR2HSV)

    # Define the green color range for masking (adjust dynamically based on image)
    lower_green = np.array([30, 80, 80])
    upper_green = np.array([100, 255, 255])

    # Create a mask to detect green color
    mask = cv2.inRange(hsv_img, lower_green, upper_green)

    # Invert the mask to keep the non-green areas (the person)
    mask_inv = cv2.bitwise_not(mask)

    # Optional: Use morphological operations to improve mask quality
    kernel = np.ones((5, 5), np.uint8)
    mask_inv = cv2.morphologyEx(mask_inv, cv2.MORPH_CLOSE, kernel)
    mask_inv = cv2.morphologyEx(mask_inv, cv2.MORPH_OPEN, kernel)

    # Display the mask for debugging
    # cv2.imshow('Mask Inverted', mask_inv)
    # cv2.waitKey(0)

    # Extract the non-green part of the image (the person)
    person = cv2.bitwise_and(padded_gs_img, padded_gs_img, mask=mask_inv)

    # Find contours of the person in the mask, focusing on external contours
    contours, hierarchy = cv2.findContours(mask_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter out small contours based on area and combine nearby contours
    filtered_contours = []
    for cnt in contours:
        if cv2.contourArea(cnt) > 100:  # Threshold area can be adjusted
            filtered_contours.append(cnt)

    if len(filtered_contours) == 0:
        print("No significant contours found!")
        return

    # Combine all filtered contours into one if needed
    combined_contour = np.vstack(filtered_contours)

    # Find the bounding box for the combined contour
    x, y, w, h = cv2.boundingRect(combined_contour)

    # If w or h is still too small, handle the error
    if w <= 1 or h <= 1:
        print("Invalid bounding box dimensions (w={}, h={})".format(w, h))
        return

    # Calculate the new position to align the person at the bottom center on the scenic background
    center_x = (scenic_width - w) // 2
    center_y = scenic_height - h  # Align at the bottom

    # Create a mask for the scenic background at the new position
    scenic_mask = np.zeros_like(scenic_img, dtype=np.uint8)
    scenic_mask[center_y:center_y+h, center_x:center_x+w] = person[y:y+h, x:x+w]

    # Display the scenic mask for debugging
    # cv2.imshow('Scenic Mask', scenic_mask)
    # cv2.waitKey(0)

    # Create the inverse of the new mask for the background
    centered_mask_inv = np.zeros_like(mask_inv)
    centered_mask_inv[center_y:center_y+h, center_x:center_x+w] = mask_inv[y:y+h, x:x+w]

    # Invert the centered mask to get the area where the person will be placed
    centered_mask = cv2.bitwise_not(centered_mask_inv)

    # Extract the part of the scenic image where the person will be placed
    scenic_background = cv2.bitwise_and(scenic_img, scenic_img, mask=centered_mask)

    # Combine the scenic background with the centered person
    final_img = cv2.add(scenic_background, scenic_mask)
    # final_img = resize_with_aspect_ratio(final_img, width=1280)

    # Display the final image
    return final_img
    # cv2.imshow('Final Image', final_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

def display_images(green_screen_img, person_with_white_bg, scenic_img, combined_img):
        # resize images to match widths
        person_with_white_bg = crop_to_match(scenic_img, person_with_white_bg)
        # print(person_with_white_bg.shape)
        green_screen_img = crop_to_match(scenic_img, green_screen_img)
        # print(person_with_white_bg.shape)
        combined_img = crop_to_match(scenic_img, combined_img)
        # print(person_with_white_bg.shape)
        scenic_img = crop_to_match(scenic_img, scenic_img)
        # print(scenic_img.shape)

        
        # concatenate images for display
        top_row = np.hstack((green_screen_img, person_with_white_bg))
        bottom_row = np.hstack((scenic_img, combined_img))
        final_display = np.vstack((top_row, bottom_row))
        final_display = resize_with_aspect_ratio(final_display, width=1680)
        # Display the images
        cv2.imshow('final display', final_display)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def TaskTwo(path_scenic_img, path_green_screen_img):
    green_screen_img = cv2.imread(path_green_screen_img)
    scenic_img = cv2.imread(path_scenic_img)
    green_screen_img_resized = cv2.resize(green_screen_img, (scenic_img.shape[1], scenic_img.shape[0]))

    # print(f"green screen image: {path_green_screen_img}, scenic image: {path_scenic_img}")


    # Extract person and create combined image
    person, person_with_white_bg, mask = extract_person(green_screen_img)
    green_screen_img_resized = crop_to_match(scenic_img, green_screen_img)
    combined_img = combined_image(path_scenic_img, path_green_screen_img)

    # Display the images
    display_images(green_screen_img_resized, person_with_white_bg, scenic_img, combined_img)

def testTaskOne():
    image_paths = [
        'scenic01.jpg',
        'scenic02.jpg',
        'scenic03.jpg',
        'scenic04.jpg',
    ]

    # color spaces
    color_spaces = [
        'CIE-XYZ',
        'CIE-Lab',
        'YCrCb',
        'HSB'
    ]

    # testing every image for every color space
    for image_path in image_paths:
        for color_space in color_spaces:
            TaskOne(color_space, image_path)

def testTaskTwo():
    green_screen_images = [
        "greenScreen01.jpg",
        "greenScreen02.jpg",
        "greenScreen03.jpg",
        "greenScreen04.jpg"
    ]

    scenic_images = [
        "scenic01.jpg",
        "scenic02.jpg",
        "scenic03.jpg",
        "scenic04.jpg",
    ]

    for path_green_screen_img in green_screen_images:
        for path_scenic_img in scenic_images:
            TaskTwo(path_scenic_img, path_green_screen_img)

def main():
    args = sys.argv[1:]
    if len(args) == 2 and args[0].startswith('-'):
        # Task One
        color_space = args[0][1:]
        image_path = args[1]

        TaskOne(color_space, image_path)
        

    elif len(args) == 2 and '.jpg' in args[0]:
        # Task Two
        scenic_img = args[0]
        green_screen_img = args[1]

        TaskTwo(green_screen_img, scenic_img)

    elif len(args) == 2 and args[0]=='test':
        print(args)
        if args[1] == 'taskone':
            testTaskOne()
        elif args[1] == 'tasktwo':
            testTaskTwo()
        else:
            print('invalid arguments')
    else:
        print("Invalid arguments.")

if __name__ == '__main__':
    main()