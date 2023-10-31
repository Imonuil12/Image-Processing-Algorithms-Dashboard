from collections import Counter
import heapq
import streamlit as st
import numpy as np
import cv2
from PIL import Image

# Histogram matching function
def hist_match(source, template):
    # Compute the CDF for the source and template images
    source_values, bin_idx, source_counts = np.unique(source, return_inverse=True, return_counts=True)
    source_cdf = np.cumsum(source_counts).astype(np.float64)
    source_cdf /= source_cdf[-1]
    
    template_values, template_counts = np.unique(template, return_counts=True)
    template_cdf = np.cumsum(template_counts).astype(np.float64)
    template_cdf /= template_cdf[-1]

    # Interpolate to find the pixel value mapping for histogram matching
    interp_values = np.interp(source_cdf, template_cdf, template_values)
    
    # Clip the values to be in the range [0, 255]
    interp_values = np.clip(interp_values, 0, 255).astype(np.uint8)
    
    return interp_values[bin_idx].reshape(source.shape)

# Noise removal algorithms
def median_filtering(img, ksize=3):
    return cv2.medianBlur(img, ksize)

def mean_filtering(img, ksize=3):
    kernel = np.ones((ksize, ksize), np.float32) / (ksize * ksize)
    return cv2.filter2D(img, -1, kernel)

def weighted_average_filtering(img):
    kernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], np.float32) / 16
    return cv2.filter2D(img, -1, kernel)

def min_filtering(img, ksize=3):
    return cv2.erode(img, np.ones((ksize, ksize)))

def max_filtering(img, ksize=3):
    return cv2.dilate(img, np.ones((ksize, ksize)))

def laplacian_enhancement(img):
    laplacian = cv2.Laplacian(img, cv2.CV_64F)
    return cv2.convertScaleAbs(img + laplacian)

# Image Resizing Algorithms
def nearest_neighbor_interpolation(img, fx=2.0, fy=2.0):
    return cv2.resize(img, None, fx=fx, fy=fy, interpolation=cv2.INTER_NEAREST)

def bilinear_interpolation(img, fx=2.0, fy=2.0):
    return cv2.resize(img, None, fx=fx, fy=fy, interpolation=cv2.INTER_LINEAR)


# Image compression Algorithms
def compute_compression_ratio(original_size, compressed_size):
    return original_size / compressed_size

def run_length_encoding(image):
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Flatten the image
    flat_gray = gray.flatten()
    # Perform RLE
    runs = []
    run_val = flat_gray[0]
    run_len = 1
    for i in range(1, len(flat_gray)):
        if flat_gray[i] == run_val:
            run_len += 1
        else:
            runs.append((run_val, run_len))
            run_val = flat_gray[i]
            run_len = 1
    runs.append((run_val, run_len))
    return runs


def region_growing_segmentation(image, seed):
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    segmented = np.zeros_like(gray, dtype=np.uint8)
    visited = np.zeros_like(gray, dtype=np.uint8)
    intensity_threshold = 20
    neighbors = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    queue = [seed]
    while queue:
        x, y = queue.pop(0)
        if 0 <= x < width and 0 <= y < height and visited[y, x] == 0:
            visited[y, x] = 1
            if abs(int(gray[y, x]) - int(gray[seed[1], seed[0]])) <= intensity_threshold:
                segmented[y, x] = 255
                for dx, dy in neighbors:
                    queue.append((x + dx, y + dy))
    return segmented


# Compression Algorithms

def arithmetic_encoding(image):
    # Convert image to 1D array
    data = image.ravel()

    # Calculate the frequency of each value in the image
    freq = Counter(data)
    total_pixels = len(data)

    # Calculate probabilities
    probs = {k: v / total_pixels for k, v in freq.items()}
    sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)

    # Calculate cumulative probabilities
    cum_prob = 0
    intervals = {}
    for value, prob in sorted_probs:
        intervals[value] = (cum_prob, cum_prob + prob)
        cum_prob += prob

    # Arithmetic encoding
    low, high = 0, 1
    for pixel in data:
        l, h = intervals[pixel]
        low, high = low + (high - low) * l, low + (high - low) * h

    # Return the average as the final code
    return (low + high) / 2


# Huffman Coding
# Huffman Node
class Node:
    def __init__(self, freq, symbol, left=None, right=None):
        self.freq = freq
        self.symbol = symbol
        self.left = left
        self.right = right
        self.huff = ''

# Huffman Coding
def calculate_frequency(image):
    # Convert image to 1D array
    values, counts = np.unique(image, return_counts=True)
    return dict(zip(values, counts))

def build_huffman_tree(freq_dict):
    nodes = [Node(freq, symbol) for symbol, freq in freq_dict.items()]
    while len(nodes) > 1:
        nodes = sorted(nodes, key=lambda x: x.freq)
        left = nodes.pop(0)
        right = nodes.pop(0)
        merged_node = Node(left.freq + right.freq, left.symbol + right.symbol, left, right)
        nodes.append(merged_node)
    return nodes[0]

def huffman_encoding(node, binary_string=''):
    if node is None:
        return {}
    if node.left is None and node.right is None:
        return {node.symbol: binary_string}
    codes = {}
    codes.update(huffman_encoding(node.left, binary_string + '0'))
    codes.update(huffman_encoding(node.right, binary_string + '1'))
    return codes

def compress_with_huffman(image):
    # Flatten the image and get unique pixel values and their counts
    pixel_values, counts = np.unique(image.ravel(), return_counts=True)
    freq = dict(zip(pixel_values, counts))

    # Create a priority queue from the frequency dictionary
    queue = [[weight, [pixel_val, ""]] for pixel_val, weight in freq.items()]
    heapq.heapify(queue)

    # Build the Huffman tree
    while len(queue) > 1:
        lo = heapq.heappop(queue)
        hi = heapq.heappop(queue)
        for pair in lo[1:]:
            pair[1] = '0' + pair[1]
        for pair in hi[1:]:
            pair[1] = '1' + pair[1]
        heapq.heappush(queue, [lo[0] + hi[0]] + lo[1:] + hi[1:])

    # Generate Huffman codes from the tree
    huff_codes = sorted(heapq.heappop(queue)[1:], key=lambda p: (len(p[-1]), p))

    # Convert the Huffman codes into a dictionary for easy lookup
    huff_dict = {str(item[0]): item[1] for item in huff_codes}

    # Compress the image using the Huffman codes
    compressed_image = ''.join([huff_dict[str(val)] for val in image.ravel()])

    return compressed_image, huff_dict

# Decompression of Huffman Code
# Decompress the Huffman-coded image
def decompress_with_huffman(compressed_image, huff_codes):
    decompressed_values = []
    temp_str = ""
    reverse_huff_codes = {v: k for k, v in huff_codes.items()}  # Reverse the Huffman codes dictionary

    for bit in compressed_image:
        temp_str += bit
        if temp_str in reverse_huff_codes:
            pixel_value = int(reverse_huff_codes[temp_str])
            decompressed_values.append(pixel_value)
            temp_str = ""

    # Convert the decompressed values back to an image
    image_size = int(len(decompressed_values)**0.5)  # Assuming the image is square
    decompressed_image = np.array(decompressed_values).reshape(image_size, image_size)
    return decompressed_image

# Image Enhancement Techniques

def image_scaling(img, scale_factor=2):
    return cv2.resize(img, None, fx=scale_factor, fy=scale_factor)

def image_negatives(img):
    return cv2.bitwise_not(img)

def log_transformations(img):
    c = 255 / np.log(1 + np.max(img))
    log_transformed = c * np.log(1 + img)
    return np.array(log_transformed, dtype=np.uint8)

def power_law_transformations(img, gamma=1.2):
    c = 255 / (np.max(img) ** gamma)
    power_law_transformed = c * (img ** gamma)
    return np.array(power_law_transformed, dtype=np.uint8)

# Contrast Stretching

def min_max_stretching(img):
    return cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)

def histogram_equalization(img):
    return cv2.equalizeHist(img)

# Image Restoration

def add_salt_and_pepper_noise(img, amount=0.02):
    noisy = np.copy(img)
    num_salt = np.ceil(amount * img.size * 0.5)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in img.shape]
    noisy[tuple(coords)] = 255
    num_pepper = np.ceil(amount * img.size * 0.5)
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in img.shape]
    noisy[tuple(coords)] = 0
    return noisy

def add_gaussian_noise(img, mean=0, sigma=25):
    row, col, ch = img.shape
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    noisy = img + gauss
    return np.clip(noisy, 0, 255).astype(np.uint8)

def add_speckle_noise(img):
    row, col, ch = img.shape
    gauss = np.random.randn(row, col, ch)
    noisy = img + img * gauss
    return np.clip(noisy, 0, 255).astype(np.uint8)

def convolution(img, kernel):
    return cv2.filter2D(img, -1, kernel)

def spatial_filtering(img, kernel_size=3):
    kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
    return cv2.filter2D(img, -1, kernel)




# Title and a brief description for your dashboard
st.title("Image Processing Dashboard. Algorithms to Enhance, Compress and Manipulate Images.")
st.write("Upload an image and select an image processing technique to apply.")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

# Dropdown menu for category selection
category = st.selectbox(
    "Choose a category",
    ["Image Enhancement Techniques", "Contrast Stretching", "Image Resizing", "Image Segmentation"]
)

if category == "Image Enhancement Techniques":
    algorithm = st.selectbox(
        "Choose an enhancement algorithm",
        ["Grayscale", "Canny Edge Detection", "Gaussian Blur", "Simple Thresholding", "Adaptive Thresholding", 
        "Watershed Segmentation", "Median Filtering", "Mean Filtering", "Weighted Average Filtering",
         "Min Filtering", "Max Filtering", "Laplacian Enhancement", "Image Scaling", "Image Negatives", "Log Transformations",
         "Power-Law Transformations", "Salt and Pepper", "Gaussian Noise", "Speckle Noise", "Convolution", "Spatial Filtering"]
    )

elif category == "Contrast Stretching":
    algorithm = st.selectbox(
        "Choose a contrast stretching algorithm",
        ["Min-Max Stretching", "Histogram Equalization", "Histogram Matching"]
    )

elif category == "Image Resizing":
    algorithm = st.selectbox(
        "Choose a resizing algorithm",
        ["Nearest Neighbor Interpolation", "Bilinear Interpolation"]
    )

elif category == "Image Segmentation":
    algorithm = st.selectbox(
        "Choose a segmentation algorithm",
        ["Region Growing Segmentation", "Thresholding", "Edge-Based Segmentation", "Region-Based Segmentation"]
    )



# The code for the algorithms

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    original_image = np.array(image)

    try:
        # Convert the image to grayscale for thresholding and segmentation methods
        gray_image = cv2.cvtColor(original_image, cv2.COLOR_RGB2GRAY)

        # Check the selected algorithm and apply the corresponding operation
        if algorithm == "Grayscale":
            st.image(gray_image, channels="GRAY", use_column_width=True)

        elif algorithm == "Canny Edge Detection":
            edges = cv2.Canny(original_image, 100, 200)
            st.image(edges, channels="GRAY", use_column_width=True)

        elif algorithm == "Gaussian Blur":
            blurred_image = cv2.GaussianBlur(original_image, (5, 5), 0)
            st.image(blurred_image, use_column_width=True)

        elif algorithm == "Simple Thresholding":
            ret, thresh = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
            st.image(thresh, channels="GRAY", use_column_width=True)

        elif algorithm == "Adaptive Thresholding":
            adaptive_thresh = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
            st.image(adaptive_thresh, channels="GRAY", use_column_width=True)

        elif algorithm == "Watershed Segmentation":
            ret, thresh = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            kernel = np.ones((3, 3), np.uint8)
            opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
            sure_bg = cv2.dilate(opening, kernel, iterations=3)
            dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
            ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
            sure_fg = np.uint8(sure_fg)
            unknown = cv2.subtract(sure_bg, sure_fg)
            ret, markers = cv2.connectedComponents(sure_fg)
            markers = markers + 1
            markers[unknown == 255] = 0
            cv2.watershed(original_image, markers)
            original_image[markers == -1] = [0, 0, 255]
            st.image(original_image, use_column_width=True)

        elif algorithm == "Median Filtering":
            filtered_image = median_filtering(gray_image)
            st.image(filtered_image, channels="GRAY", use_column_width=True)

        elif algorithm == "Mean Filtering":
            filtered_image = mean_filtering(gray_image)
            st.image(filtered_image, channels="GRAY", use_column_width=True)

        elif algorithm == "Weighted Average Filtering":
            filtered_image = weighted_average_filtering(gray_image)
            st.image(filtered_image, channels="GRAY", use_column_width=True)

        elif algorithm == "Min Filtering":
            filtered_image = min_filtering(gray_image)
            st.image(filtered_image, channels="GRAY", use_column_width=True)

        elif algorithm == "Max Filtering":
            filtered_image = max_filtering(gray_image)
            st.image(filtered_image, channels="GRAY", use_column_width=True)

        elif algorithm == "Laplacian Enhancement":
            enhanced_image = laplacian_enhancement(gray_image)
            st.image(enhanced_image, channels="GRAY", use_column_width=True)

        elif algorithm == "Nearest Neighbor Interpolation":
            resized_image = nearest_neighbor_interpolation(original_image)
            st.image(resized_image, use_column_width=True)

        elif algorithm == "Bilinear Interpolation":
            resized_image = bilinear_interpolation(original_image)
            st.image(resized_image, use_column_width=True)

        elif algorithm == "Run Length Encoding":
            runs = run_length_encoding(original_image)
            st.write(f"Run Length Encoding: {runs[:10]}...")  # Displaying only the first 10 runs for brevity

        elif algorithm == "Region Growing Segmentation":
            seed = (gray_image.shape[1] // 2, gray_image.shape[0] // 2)  # Using the center pixel as seed
            segmented_image = region_growing_segmentation(original_image, seed)
            st.image(segmented_image, channels="GRAY", use_column_width=True)

        elif algorithm == "Huffman Coding":
            compressed_image, huff_codes = compress_with_huffman(gray_image)
            st.write(f"Compressed Image: {compressed_image[:500]}...")  # Displaying only the first 500 characters for brevity
            st.write(f"Huffman Codes: {huff_codes}")

            # Add a button for decoding
            if st.button('Decode'):
                decompressed_image = decompress_with_huffman(compressed_image, huff_codes)
                st.image(decompressed_image, caption="Decompressed Image", use_column_width=True)

        elif algorithm == "Arithmetic Encoding":
            encoded_value = arithmetic_encoding(gray_image)
            st.write(f"Encoded Value: {encoded_value}")
            
        elif algorithm == "Image Scaling":
            scaled_image = image_scaling(original_image)
            st.image(scaled_image, use_column_width=True)

        elif algorithm == "Image Negatives":
            negative_image = image_negatives(original_image)
            st.image(negative_image, use_column_width=True)

        elif algorithm == "Log Transformations":
            log_transformed_image = log_transformations(gray_image)
            st.image(log_transformed_image, channels="GRAY", use_column_width=True)

        elif algorithm == "Power-Law Transformations":
            power_law_image = power_law_transformations(gray_image)
            st.image(power_law_image, channels="GRAY", use_column_width=True)

        elif algorithm == "Min-Max Stretching":
            min_max_image = min_max_stretching(gray_image)
            st.image(min_max_image, channels="GRAY", use_column_width=True)

        elif algorithm == "Histogram Equalization":
            hist_eq_image = histogram_equalization(gray_image)
            st.image(hist_eq_image, channels="GRAY", use_column_width=True)

        elif algorithm == "Salt and Pepper":
            salt_pepper_image = add_salt_and_pepper_noise(original_image)
            st.image(salt_pepper_image, use_column_width=True)

        elif algorithm == "Gaussian Noise":
            gaussian_noise_image = add_gaussian_noise(original_image)
            st.image(gaussian_noise_image, use_column_width=True)

        elif algorithm == "Speckle Noise":
            speckle_noise_image = add_speckle_noise(original_image)
            st.image(speckle_noise_image, use_column_width=True)

        elif algorithm == "Convolution":
            # Example kernel for convolution (you can change it)
            kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
            convoluted_image = convolution(original_image, kernel)
            st.image(convoluted_image, use_column_width=True)

        elif algorithm == "Spatial Filtering":
            spatial_filtered_image = spatial_filtering(gray_image)
            st.image(spatial_filtered_image, channels="GRAY", use_column_width=True)

        elif algorithm == "Histogram Matching":
                template_file = st.file_uploader("Choose a template image for histogram matching...", type=["jpg", "png", "jpeg"])
            
                if template_file:
                    template_image = Image.open(template_file)
                    st.image(template_image, caption='Template Image.', use_column_width=True)
                    template_image_array = np.array(template_image)
                    
                    # Convert the template image to grayscale
                    template_gray = cv2.cvtColor(template_image_array, cv2.COLOR_RGB2GRAY)
                    
                    # Apply histogram matching
                    matched_image = hist_match(gray_image, template_gray)
                    st.image(matched_image, channels="GRAY", use_column_width=True)
                else:
                    st.warning("Please upload a template image for histogram matching.")
                
    
        elif algorithm == "Thresholding":
            ret, thresh = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            st.image(thresh, channels="GRAY", use_column_width=True)

        elif algorithm == "Edge-Based Segmentation":
            edges = cv2.Canny(gray_image, 100, 200)
            st.image(edges, channels="GRAY", use_column_width=True)

        elif algorithm == "Region-Based Segmentation":
            seed = (gray_image.shape[1] // 2, gray_image.shape[0] // 2)  # Using the center pixel as seed
            segmented_image = region_growing_segmentation(original_image, seed)
            st.image(segmented_image, channels="GRAY", use_column_width=True)


    except Exception as e:
        st.error(f"Error in {algorithm}: {e}")

else:
    st.warning("Please upload an image.")