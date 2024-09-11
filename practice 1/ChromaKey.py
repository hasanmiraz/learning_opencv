import sys
import cv2
import numpy as np

# to crop and resize image_b to match the aspect ratio and size of image_a
def crop_to_match(image_a, image_b):
    a_height, a_width = image_a.shape[:2]
    b_height, b_width = image_b.shape[:2]

    aspect_a = a_width / a_height
    aspect_b = b_width / b_height

    # crop image_b horizontally or vertically based on the aspect ratio
    if aspect_b > aspect_a:
        new_width = int(b_height * aspect_a)
        start_x = (b_width - new_width) // 2
        cropped_b = image_b[:, start_x:start_x + new_width]
    else:
        new_height = int(b_width / aspect_a)
        start_y = (b_height - new_height) // 2
        cropped_b = image_b[start_y:start_y + new_height, :]

    # resize the cropped image to match the dimensions of image_a
    resized_b = cv2.resize(cropped_b, (a_width, a_height), interpolation=cv2.INTER_AREA)

    return resized_b

# to resize an image while maintaining its aspect ratio
def resize_with_aspect_ratio(image, width=None, height=None, inter=cv2.INTER_AREA):
    (h, w) = image.shape[:2]

    # If both width and height are None, return the original image
    if width is None and height is None:
        return image

    # Calculate new dimensions based on the provided width or height
    if width is None:
        aspect_ratio = height / float(h)
        new_dimensions = (int(w * aspect_ratio), height)
    else:
        aspect_ratio = width / float(w)
        new_dimensions = (width, int(h * aspect_ratio))

    # resize the image to the new dimensions
    resized_image = cv2.resize(image, new_dimensions, interpolation=inter)
    return resized_image

# to resize img2 to match half the width of img1
def resize_to_match_width(img1, img2):
    resized_img2 = cv2.resize(img2, (int(img1.shape[1] / 2), int(img1.shape[0] / 2)))
    return resized_img2

# TaskOne: convert an image to a specified color space and display various channels
def TaskOne(color_space, image_path):
    print(f'task1: {color_space}, {image_path}')
    image = cv2.imread(image_path)
    # height, width, channels = image.shape

    # Resize the image to half the width of 1280 pixels
    image = resize_with_aspect_ratio(image, width=(1280 // 2))
    # image = cv2.resize(image, (int(width/2), int(height/2)))

    # Convert the image to the specified color space
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

    # Split the converted image into three channels
    C1, C2, C3 = cv2.split(converted_image)

    # Normalize the channels to the range [0, 255]
    C1 = cv2.normalize(C1, None, 0, 255, cv2.NORM_MINMAX)
    C2 = cv2.normalize(C2, None, 0, 255, cv2.NORM_MINMAX)
    C3 = cv2.normalize(C3, None, 0, 255, cv2.NORM_MINMAX)

    # Convert the grayscale channels back to BGR format for display
    C1 = cv2.cvtColor(C1, cv2.COLOR_GRAY2BGR)
    C2 = cv2.cvtColor(C2, cv2.COLOR_GRAY2BGR)
    C3 = cv2.cvtColor(C3, cv2.COLOR_GRAY2BGR)

    # Create a final image with the original and channel images arranged in a grid
    top_row = np.hstack((image, C2))
    bottom_row = np.hstack((C1, C3))
    final_image = np.vstack((top_row, bottom_row))

    # Display the final image
    cv2.imshow('task 1', final_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# to extract a person from a green screen image by removing the green background
def extract_person(green_screen_img):
    hsv = cv2.cvtColor(green_screen_img, cv2.COLOR_BGR2HSV)
    
    # define the range for green color in HSV
    lower_green = np.array([30, 90, 90])
    upper_green = np.array([100, 255, 255])
    
    # Create a mask to isolate the green background
    mask = cv2.inRange(hsv, lower_green, upper_green)
    
    # Invert the mask to keep the person and remove the background
    mask_inv = cv2.bitwise_not(mask)
    
    # extract the person from the image using the inverted mask
    person = cv2.bitwise_and(green_screen_img, green_screen_img, mask=mask_inv)
    
    # Create a white background image
    white_bg = np.ones_like(green_screen_img) * 255
    
    # combine the extracted person with the white background
    person_with_white_bg = cv2.add(person, cv2.bitwise_and(white_bg, white_bg, mask=mask))
    
    return person, person_with_white_bg, mask

# to combine a green screen image with a scenic background
def combined_image(path_scenic_img, path_green_screen_img):
    green_screen_img = cv2.imread(path_green_screen_img)
    scenic_img = cv2.imread(path_scenic_img)

    gs_height, gs_width = green_screen_img.shape[:2]
    scenic_height, scenic_width = scenic_img.shape[:2]

    # Calculate padding to center the green screen image on the scenic background
    pad_top = scenic_height - gs_height
    pad_bottom = 0
    pad_left = (scenic_width - gs_width) // 2
    pad_right = scenic_width - gs_width - pad_left

    # add padding to the green screen image
    padded_gs_img = cv2.copyMakeBorder(
        green_screen_img,
        pad_top, pad_bottom, pad_left, pad_right,
        cv2.BORDER_CONSTANT,
        value=[0, 255, 0]  
    )

    # Convert the padded green screen image to HSV color space
    hsv_img = cv2.cvtColor(padded_gs_img, cv2.COLOR_BGR2HSV)

    # Define the range for green color in HSV
    lower_green = np.array([30, 80, 80])
    upper_green = np.array([100, 255, 255])

    # Create a mask to isolate the green background
    mask = cv2.inRange(hsv_img, lower_green, upper_green)

    # Invert the mask to keep the person and remove the background
    mask_inv = cv2.bitwise_not(mask)

    # apply morphological operations to clean up the mask
    kernel = np.ones((5, 5), np.uint8)
    mask_inv = cv2.morphologyEx(mask_inv, cv2.MORPH_CLOSE, kernel)
    mask_inv = cv2.morphologyEx(mask_inv, cv2.MORPH_OPEN, kernel)

    # Display the mask for debugging
    # cv2.imshow('Mask Inverted', mask_inv)
    # cv2.waitKey(0)

    # extract the person from the padded green screen image using the inverted mask
    person = cv2.bitwise_and(padded_gs_img, padded_gs_img, mask=mask_inv)

    # Find contours in the inverted mask
    contours, hierarchy = cv2.findContours(mask_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter out small contours based on area
    filtered_contours = []
    for cnt in contours:
        if cv2.contourArea(cnt) > 100:
            filtered_contours.append(cnt)

    # check if any significant contours were found
    if len(filtered_contours) == 0:
        print("No significant contours found!")
        return

    # Combine all significant contours into one
    combined_contour = np.vstack(filtered_contours)

    # Get the bounding box of the combined contour
    x, y, w, h = cv2.boundingRect(combined_contour)

    # Check if the bounding box dimensions are valid
    if w <= 1 or h <= 1:
        print("Invalid bounding box dimensions (w={}, h={})".format(w, h))
        return

    # Calculate the center position for the person on the scenic background
    center_x = (scenic_width - w) // 2
    center_y = scenic_height - h

    # Create a mask for the person on the scenic background
    scenic_mask = np.zeros_like(scenic_img, dtype=np.uint8)
    scenic_mask[center_y:center_y+h, center_x:center_x+w] = person[y:y+h, x:x+w]

    # display the scenic mask for debugging
    # cv2.imshow('Scenic Mask', scenic_mask)
    # cv2.waitKey(0)

    # center the inverted mask on the scenic background
    centered_mask_inv = np.zeros_like(mask_inv)
    centered_mask_inv[center_y:center_y+h, center_x:center_x+w] = mask_inv[y:y+h, x:x+w]

    # invert the centered mask
    centered_mask = cv2.bitwise_not(centered_mask_inv)

    #extract the scenic background excluding the area where the person will be placed
    scenic_background = cv2.bitwise_and(scenic_img, scenic_img, mask=centered_mask)

    # combine the scenic background with the person
    final_img = cv2.add(scenic_background, scenic_mask)
    # final_img = resize_with_aspect_ratio(final_img, width=1280)

    return final_img
    # cv2.imshow('Final Image', final_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

# to display images side by side for comparison
def display_images(green_screen_img, person_with_white_bg, scenic_img, combined_img):
    # Crop the images to match the size and aspect ratio of the scenic image
    person_with_white_bg = crop_to_match(scenic_img, person_with_white_bg)
    # print(person_with_white_bg.shape)
    green_screen_img = crop_to_match(scenic_img, green_screen_img)
    # print(green_screen_img.shape)
    combined_img = crop_to_match(scenic_img, combined_img)
    # print(combined_img.shape)
    scenic_img = crop_to_match(scenic_img, scenic_img)
    # print(scenic_img.shape)

    # Arrange the images in a grid
    top_row = np.hstack((green_screen_img, person_with_white_bg))
    bottom_row = np.hstack((scenic_img, combined_img))
    final_display = np.vstack((top_row, bottom_row))

    # Resize the final display image to fit within the specified width
    final_display = resize_with_aspect_ratio(final_display, width=1680)

    # Display the final combined image
    cv2.imshow('final display', final_display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# TaskTwo: Combine green screen and scenic images and display the results
def TaskTwo(path_scenic_img, path_green_screen_img):
    green_screen_img = cv2.imread(path_green_screen_img)
    scenic_img = cv2.imread(path_scenic_img)

    # Resize the green screen image to match the dimensions of the scenic image
    green_screen_img_resized = cv2.resize(green_screen_img, (scenic_img.shape[1], scenic_img.shape[0]))

    print(f"GSImage: {path_green_screen_img}, SImage: {path_scenic_img}")

    # Extract the person from the green screen image
    person, person_with_white_bg, mask = extract_person(green_screen_img)

    # Crop and resize the green screen image to match the scenic image
    green_screen_img_resized = crop_to_match(scenic_img, green_screen_img)

    # Combine the green screen image with the scenic background
    combined_img = combined_image(path_scenic_img, path_green_screen_img)

    # Display the images side by side
    display_images(green_screen_img_resized, person_with_white_bg, scenic_img, combined_img)

# Test for TaskOne
def testTaskOne():
    image_paths = [
        'scenic01.jpg',
        'scenic02.jpg',
        'scenic03.jpg',
        'scenic04.jpg',
    ]

    color_spaces = [
        'CIE-XYZ',
        'CIE-Lab',
        'YCrCb',
        'HSB'
    ]

    # Iterate over image paths and color spaces to test TaskOne
    for image_path in image_paths:
        for color_space in color_spaces:
            TaskOne(color_space, image_path)

# Test for TaskTwo
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

    # Iterate over green screen and scenic images to test TaskTwo
    for path_green_screen_img in green_screen_images:
        for path_scenic_img in scenic_images:
            TaskTwo(path_scenic_img, path_green_screen_img)

# Main function to handle command-line arguments and execute tasks
def main():
    args = sys.argv[1:]
    
    # If two arguments are provided with the first being a flag, execute TaskOne
    if len(args) == 2 and args[0].startswith('-'):
        color_space = args[0][1:]
        image_path = args[1]
        TaskOne(color_space, image_path)
        
    # If two image paths are provided, execute TaskTwo
    elif len(args) == 2 and '.jpg' in args[0]:
        scenic_img = args[0]
        green_screen_img = args[1]
        TaskTwo(scenic_img, green_screen_img)

    # If "test" is the first argument, run the corresponding test function
    elif len(args) == 2 and args[0] == 'test':
        if args[1] == 'taskone':
            testTaskOne()
        elif args[1] == 'tasktwo':
            testTaskTwo()
        else:
            print('Invalid arguments')
    else:
        print("Invalid arguments.")

# Entry point of the script
if __name__ == '__main__':
    main()
