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




# Title and a brief description for your dashboard
st.title("Image Processing Dashboard")
st.write("Upload an image and select an image processing technique to apply.")

# Allow users to upload an image using Streamlit's st.file_uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    original_image = np.array(image)


    # Grayscale
    if st.button('Grayscale', key='btn_grayscale'):
        gray_image = cv2.cvtColor(original_image, cv2.COLOR_RGB2GRAY)
        st.image(gray_image, channels="GRAY", use_column_width=True)


    # Canny Edge Detection
    if st.button('Canny Edge Detection', key='btn_canny'):
        edges = cv2.Canny(original_image, 100, 200)
        st.image(edges, channels="GRAY", use_column_width=True)


    # Gaussian Blur
    if st.button('Gaussian Blur', key='btn_gaussian_blur'):
        blurred_image = cv2.GaussianBlur(original_image, (15, 15), 0)
        st.image(blurred_image, channels="RGB", use_column_width=True)


    # Min-Max Stretching
    if st.button('Min-Max Stretching', key='btn_min_max'):
        gray_image = cv2.cvtColor(original_image, cv2.COLOR_RGB2GRAY)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(gray_image)
        min_max_image = (255 / (max_val - min_val)) * (gray_image - min_val)
        st.image(min_max_image.astype(np.uint8), channels="GRAY", use_column_width=True)


    # Histogram Equalization
    if st.button('Histogram Equalization', key='btn_hist_eq'):
        gray_image = cv2.cvtColor(original_image, cv2.COLOR_RGB2GRAY)
        equalized_image = cv2.equalizeHist(gray_image)
        st.image(equalized_image, channels="GRAY", use_column_width=True)


    # Histogram Matching
    reference_file = st.file_uploader("Choose a reference image for Histogram Matching...", type=["jpg", "png", "jpeg"])
    if reference_file is not None:
        reference_image = Image.open(reference_file)
        st.image(reference_image, caption='Reference Image.', use_column_width=True)
        reference_image_np = np.array(reference_image)
        if st.button('Histogram Matching', key='btn_hist_match'):
            gray_image = cv2.cvtColor(original_image, cv2.COLOR_RGB2GRAY)
            gray_reference = cv2.cvtColor(reference_image_np, cv2.COLOR_RGB2GRAY)
            matched_image = hist_match(gray_image, gray_reference)
            # Clamp the values to [0, 255] and convert to np.uint8
            matched_image = np.clip(matched_image, 0, 255).astype(np.uint8)
            st.image(matched_image, channels="GRAY", use_column_width=True)


    # Image Matching
    target_file = st.file_uploader("Choose a target image for Image Matching...", type=["jpg", "png", "jpeg"])
    if target_file is not None:
        target_image = Image.open(target_file)
        st.image(target_image, caption='Target Image.', use_column_width=True)
        target_image_np = np.array(target_image)
        if st.button('Image Matching', key='btn_img_match'):
            # Convert images to grayscale
            gray_image = cv2.cvtColor(original_image, cv2.COLOR_RGB2GRAY)
            gray_target = cv2.cvtColor(target_image_np, cv2.COLOR_RGB2GRAY)
            # Initialize ORB detector
            orb = cv2.ORB_create()
            # Find the keypoints and descriptors with ORB
            kp1, des1 = orb.detectAndCompute(gray_image, None)
            kp2, des2 = orb.detectAndCompute(gray_target, None)
            # Create BFMatcher object
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            # Match descriptors
            matches = bf.match(des1, des2)
            # Sort them in ascending order of distance
            matches = sorted(matches, key=lambda x: x.distance)
            # Draw first 50 matches
            img_matches = cv2.drawMatches(gray_image, kp1, gray_target, kp2, matches[:50], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            st.image(img_matches, channels="BGR", use_column_width=True)


    # Image Restoration
    if 'restoration_button_pressed' not in st.session_state:
        st.session_state.restoration_button_pressed = False
    if st.button('Image Restoration', key='btn_img_restoration'):
        st.session_state.restoration_button_pressed = True
    if st.session_state.restoration_button_pressed:
        if len(original_image.shape) == 3 and original_image.shape[2] == 3:  # Colored image
            restored_image = cv2.fastNlMeansDenoisingColored(original_image, None, 10, 10, 7, 21)
            st.image(restored_image, channels="BGR", use_column_width=True)
        else:  # Grayscale image
            restored_image = cv2.fastNlMeansDenoising(original_image, None, 30, 7, 21)
            st.image(restored_image, channels="GRAY", use_column_width=True)
        st.session_state.restoration_button_pressed = False  # Reset the button state


    # Image Thresholding
    if st.button('Simple Thresholding', key='btn_simple_threshold'):
        gray_image = cv2.cvtColor(original_image, cv2.COLOR_RGB2GRAY)
        ret, thresh1 = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
        st.image(thresh1, channels="GRAY", use_column_width=True)
    if st.button('Adaptive Thresholding', key='btn_adaptive_threshold'):
        gray_image = cv2.cvtColor(original_image, cv2.COLOR_RGB2GRAY)
        thresh2 = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
        st.image(thresh2, channels="GRAY", use_column_width=True)
    if st.button('Otsu’s Binarization', key='btn_otsu_threshold'):
        gray_image = cv2.cvtColor(original_image, cv2.COLOR_RGB2GRAY)
        ret, thresh3 = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        st.image(thresh3, channels="GRAY", use_column_width=True)


    # Image Segmentation using Watershed Algorithm
    if st.button('Watershed Segmentation', key='btn_watershed'):
        gray_image = cv2.cvtColor(original_image, cv2.COLOR_RGB2GRAY)
        # Apply binary threshold
        ret, thresh = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        # Noise removal using Morphological closing operation
        kernel = np.ones((3,3),np.uint8)
        closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
        # Sure background area
        sure_bg = cv2.dilate(closing, kernel, iterations=3)
        # Finding sure foreground area
        dist_transform = cv2.distanceTransform(closing, cv2.DIST_L2, 5)
        ret, sure_fg = cv2.threshold(dist_transform, 0.7*dist_transform.max(), 255, 0)
        # Finding unknown region
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg, sure_fg)
        # Marker labelling
        ret, markers = cv2.connectedComponents(sure_fg)
        # Add one to all the labels to distinguish them from the background
        markers = markers + 1
        markers[unknown == 255] = 0
        # Apply watershed
        cv2.watershed(original_image, markers)
        original_image[markers == -1] = [0, 0, 255]  # Mark the boundary region in red
        st.image(original_image, channels="BGR", use_column_width=True)


    # Noise Removal Algorithms Section
    st.subheader("Noise Removal Algorithms")
    if st.button('Median Filtering', key='btn_median_filtering'):
        filtered_image = median_filtering(original_image)
        st.image(filtered_image, channels="BGR", use_column_width=True)

    if st.button('Mean Filtering', key='btn_mean_filtering'):
        filtered_image = mean_filtering(original_image)
        st.image(filtered_image, channels="BGR", use_column_width=True)

    if st.button('Weighted Average Filtering', key='btn_weighted_avg_filtering'):
        filtered_image = weighted_average_filtering(original_image)
        st.image(filtered_image, channels="BGR", use_column_width=True)

    if st.button('Min Filtering', key='btn_min_filtering'):
        filtered_image = min_filtering(original_image)
        st.image(filtered_image, channels="BGR", use_column_width=True)

    if st.button('Max Filtering', key='btn_max_filtering'):
        filtered_image = max_filtering(original_image)
        st.image(filtered_image, channels="BGR", use_column_width=True)

    if st.button('Laplacian Enhancement', key='btn_laplacian_enhancement'):
        enhanced_image = laplacian_enhancement(original_image)
        st.image(enhanced_image, channels="BGR", use_column_width=True)


    # Image Resizing Algorithms Section
    st.subheader("Image Resizing Algorithms")
    if st.button('Nearest Neighbor Interpolation', key='btn_nearest_neighbor'):
        resized_image = nearest_neighbor_interpolation(original_image)
        st.image(resized_image, channels="BGR", use_column_width=True)

    if st.button('Bilinear Interpolation', key='btn_bilinear_interpolation'):
        resized_image = bilinear_interpolation(original_image)
        st.image(resized_image, channels="BGR", use_column_width=True)

    # Image compression Algorithms
    if st.button('Run Length Encoding', key='btn_rle'):
        rle_result = run_length_encoding(original_image)
        st.write(f"RLE Result: {rle_result}")


    # Region growing segmentation
    seed_x = st.slider('Seed X', 0, original_image.shape[1] - 1, int(original_image.shape[1] / 2))
    seed_y = st.slider('Seed Y', 0, original_image.shape[0] - 1, int(original_image.shape[0] / 2))
    if st.button('Region Growing Segmentation', key='btn_region_growing'):
        segmented_image = region_growing_segmentation(original_image, (seed_x, seed_y))
        st.image(segmented_image, channels="GRAY", use_column_width=True)


    # Huffman Coding
    if st.button('Huffman Coding', key='btn_huffman'):
        compressed_image, huff_codes = compress_with_huffman(original_image)
        if compressed_image:
            st.write(f"Compressed Image: {compressed_image[:500]}...")  # Displaying only the first 500 characters for brevity
            st.write(f"Huffman Codes: {huff_codes}")
        else:
            st.warning("Failed to compress the image using Huffman Coding. Please ensure an image is uploaded.")

    # Decompression of Huffman Coding
    
    if cv2.Algorithm == "Huffman Coding":
            compressed_image, huff_codes = compress_with_huffman(original_image)
        
        # Check if compressed_image exists before displaying
            if 'compressed_image' in locals():
                st.write(f"Compressed Image: {compressed_image[:500]}...")  # Displaying only the first 500 characters for brevity
                st.write(f"Huffman Codes: {huff_codes}")

                # Add a button for decoding
                if st.button('Decode'):
                    # Decompress the image
                    decompressed_image = decompress_with_huffman(compressed_image, huff_codes)
                    st.image(decompressed_image, caption="Decompressed Image", use_column_width=True)


else:
    st.warning("Please upload an image.")
