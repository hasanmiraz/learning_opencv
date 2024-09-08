import cv2
import numpy as np

class TaskOne:
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

    def display_image_components(self, image_path, color_space, window_name):
        print(f'task1: {color_space}, {image_path}')
        # reading the image and resizing it
        image = cv2.imread(image_path)
        height, width, channels = image.shape
        image = self.resize_with_aspect_ratio(image, width = (1280//2))
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
        cv2.imshow(window_name, final_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def test(self):
        # image paths
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
                self.process(image_path, color_space)
                # try:
                #     display_image_components(image_path, color_space, f'{image_path.replace(".jpg", "")}_{color_space}')
                #     print(f"Test passed for {image_path} with color space {color_space}")
                # except Exception as e:
                #     print(f"Test failed for {image_path} with color space {color_space}: {e}")

    def process(self, image_path, color_space):
        try:
            self.display_image_components(image_path, color_space, f'{image_path.replace(".jpg", "")}_{color_space}')
            print(f"Test passed for {image_path} with color space {color_space}")
        except Exception as e:
            print(f"Test failed for {image_path} with color space {color_space}: {e}")