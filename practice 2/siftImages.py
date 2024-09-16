import cv2
import numpy as np
import matplotlib.pyplot as plt
import itertools

def rescale_image(image, target_width=600, target_height=480):
    height, width = image.shape[:2]
    scaling_factor = min(target_width / width, target_height / height)
    new_size = (int(width * scaling_factor), int(height * scaling_factor))
    return cv2.resize(image, new_size)

def extract_keypoints(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to luminance (grayscale)
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    return keypoints, descriptors

def draw_keypoints(image, keypoints):
    for kp in keypoints:
        x, y = int(kp.pt[0]), int(kp.pt[1])
        scale = int(kp.size / 2)
        angle = kp.angle
        end_x = int(x + scale * np.cos(angle * np.pi / 180.0))
        end_y = int(y - scale * np.sin(angle * np.pi / 180.0))

        cv2.circle(image, (x, y), scale, (0, 255, 0), 1)
        cv2.line(image, (x, y), (end_x, end_y), (255, 0, 0), 1)
        cv2.drawMarker(image, (x, y), (0, 0, 255), cv2.MARKER_CROSS, 10, 1)
    return image

def display_keypoints(image, keypoints):
    image_with_keypoints = draw_keypoints(image.copy(), keypoints)
    combined_image = np.hstack((image, image_with_keypoints))
    plt.imshow(cv2.cvtColor(combined_image, cv2.COLOR_BGR2RGB))
    plt.title(f'Detected Keypoints: {len(keypoints)}')
    plt.axis('off')
    plt.show()

def kmeans_clustering(descriptors, k, max_iterations=100):
    centroids = descriptors[np.random.choice(descriptors.shape[0], k, replace=False)]
    for _ in range(max_iterations):
        distances = np.linalg.norm(descriptors[:, np.newaxis] - centroids, axis=2)
        closest_clusters = np.argmin(distances, axis=1)

        new_centroids = np.array([descriptors[closest_clusters == i].mean(axis=0) for i in range(k)])
        valid_centroids = ~np.isnan(new_centroids).any(axis=1)
        centroids[valid_centroids] = new_centroids[valid_centroids]

        if np.all(centroids == new_centroids):
            break

    return centroids, closest_clusters

def create_histogram(descriptors, centroids):
    histogram = np.zeros(len(centroids))
    distances = np.linalg.norm(descriptors[:, np.newaxis] - centroids, axis=2)
    closest_clusters = np.argmin(distances, axis=1)
    for cluster in closest_clusters:
        histogram[cluster] += 1
    return histogram

def calculate_chi_square_distance(hist1, hist2):
    return 0.5 * np.sum((hist1 - hist2) ** 2 / (hist1 + hist2 + 1e-10))

def compare_images(images, file_names, k_percentages=[5, 10, 20]):
    all_descriptors = []
    keypoints_list = []
    for image in images:
        keypoints, descriptors = extract_keypoints(image)
        keypoints_list.append(keypoints)
        if descriptors is not None:
            all_descriptors.extend(descriptors)

    total_keypoints = len(all_descriptors)
    print(f"Total Keypoints: {total_keypoints}")

    for k_percentage in k_percentages:
        k = max(1, int(total_keypoints * (k_percentage / 100)))
        print(f"\nClustering with K = {k} ({k_percentage}% of total keypoints)")
        centroids, _ = kmeans_clustering(np.array(all_descriptors), k)

        histograms = [create_histogram(d, centroids) if d is not None else np.zeros(k) for d in
                      [extract_keypoints(img)[1] for img in images]]

        dissimilarity_matrix = np.zeros((len(images), len(images)))

        for i in range(len(histograms)):
            for j in range(i, len(histograms)):
                distance = calculate_chi_square_distance(histograms[i], histograms[j])
                dissimilarity_matrix[i][j] = distance
                dissimilarity_matrix[j][i] = distance  # Symmetric

        # Print matrix in a readable format using string formatting
        print(f"\nDissimilarity Matrix for K={k} ({k_percentage}%):")
        
        # Print header
        gap_len = " "*len(file_names[0])
        header = gap_len+"".join([f"{file_names[i]:^20}" for i in range(len(images))])
        print(header)
        
        # Print each row
        for i in range(len(images)):
            row = f"{file_names[i]}" + "".join([f"{dissimilarity_matrix[i][j]:^20.4f}" for j in range(len(images))])
            print(row)

if __name__ == "__main__":
    import sys
    file_names = [filename for filename in sys.argv[1:]]
    input_images = [cv2.imread(filename) for filename in sys.argv[1:]]
    rescaled_images = [rescale_image(image) for image in input_images]

    if len(rescaled_images) == 1:
        # Task One: Single image
        keypoints, _ = extract_keypoints(rescaled_images[0])
        display_keypoints(rescaled_images[0], keypoints)
        print(f"Image 1: {len(keypoints)} keypoints detected.")
    elif len(rescaled_images) > 1:
        # Task Two: Multiple images
        for idx, image in enumerate(rescaled_images):
            keypoints, _ = extract_keypoints(image)
            print(f"Image {file_names[idx]}: {len(keypoints)} keypoints detected.")
        compare_images(rescaled_images, file_names)
    else:
        file_names = []
        for i in range(1, 16):
            file_names.append(f'img{i:02}.JPG')
        current = 0
        combinations = list(set(itertools.combinations(file_names, 5)))

        for combination in combinations:
            print(f'currently executing combination: {current}:')
            print(combination)
            input_images = [cv2.imread(filename) for filename in combinations]
            rescaled_images = [rescale_image(image) for image in input_images]

            for idx, image in enumerate(rescaled_images):
                keypoints, _ = extract_keypoints(image)
                print(f"Image {file_names[idx]}: {len(keypoints)} keypoints detected.")
            compare_images(rescaled_images, combination)
