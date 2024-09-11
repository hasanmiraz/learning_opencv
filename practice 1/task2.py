import cv2
import numpy as np

class TaskTwo:
    def extract_person(self, green_screen_img):
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
        

    def combine_images(self, person, scenic_img, mask):
        # Resize person and mask to match the scenic image size
        person_resized = cv2.resize(person, (scenic_img.shape[1], scenic_img.shape[0]))
        mask_resized = cv2.resize(mask, (scenic_img.shape[1], scenic_img.shape[0]))
        
        # Align the person to the bottom center of the scenic image
        y_offset = scenic_img.shape[0] - person_resized.shape[0]
        x_offset = (scenic_img.shape[1] - person_resized.shape[1]) // 2
        
        # Create a region of interest (ROI) in the scenic image
        roi = scenic_img[y_offset:y_offset + person_resized.shape[0], x_offset:x_offset + person_resized.shape[1]]
        
        # Combine the person with the scenic image
        scenic_bg = cv2.bitwise_and(roi, roi, mask=mask_resized)
        combined = cv2.add(scenic_bg, person_resized)
        
        return combined

    def resize_to_match_width(self, img1, img2):
        # resize img2 to match the width of img1
        resized_img2 = cv2.resize(img2, (int(img1.shape[1]/2), int(img1.shape[0]/2)))
        return resized_img2

    def display_images(self, green_screen_img, person_with_white_bg, scenic_img, combined_img):
        original = scenic_img
        # resize images to match widths
        person_with_white_bg = self.resize_to_match_width(scenic_img, person_with_white_bg)
        print(person_with_white_bg.shape)
        green_screen_img = self.resize_to_match_width(scenic_img, green_screen_img)
        print(person_with_white_bg.shape)
        combined_img = self.resize_to_match_width(scenic_img, combined_img)
        print(person_with_white_bg.shape)
        scenic_img = self.resize_to_match_width(scenic_img, scenic_img)
        print(scenic_img.shape)

        
        # concatenate images for display
        top_row = np.hstack((green_screen_img, person_with_white_bg))
        bottom_row = np.hstack((scenic_img, combined_img))
        final_display = np.vstack((top_row, bottom_row))
         
        # Display the images
        cv2.imshow('final display', final_display)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # load images
    def test(self):
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
                self.process(path_green_screen_img, path_scenic_img)
                # green_screen_img = cv2.imread(path_green_screen_img)
                # scenic_img = cv2.imread(path_scenic_img)
                # print(f"green screen image: {path_green_screen_img}, scenic image: {path_scenic_img}")
                # scenic_img = resize_to_match_width(green_screen_img, scenic_img)


                # # Extract person and create combined image
                # person, person_with_white_bg, mask = extract_person(green_screen_img)
                # combined_img = combine_images(person, scenic_img, mask)

                # # Display the images
                # display_images(green_screen_img, person_with_white_bg, scenic_img, combined_img)

    def process(self, path_green_screen_img, path_scenic_img):
        green_screen_img = cv2.imread(path_green_screen_img)
        scenic_img = cv2.imread(path_scenic_img)

        height, width, channels = scenic_img.shape
        print(f'width: {width}')
        if width < 1280:
            width = 1280 
        elif width > 1680:
            width = 1680

        scenic_img = self.resize_with_aspect_ratio(scenic_img, width=width)

        print(f"green screen image: {path_green_screen_img}, scenic image: {path_scenic_img}")


        # Extract person and create combined image
        person, person_with_white_bg, mask = self.extract_person(green_screen_img)
        green_screen_img = self.resize_to_match_width(scenic_img, green_screen_img)
        combined_img = self.combine_images(person, scenic_img, mask)

        # Display the images
        self.display_images(green_screen_img, person_with_white_bg, scenic_img, combined_img)

    def resize_with_aspect_ratio(self, image, width=None, height=None, inter=cv2.INTER_AREA):
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
    
obj = TaskTwo()
obj.test()