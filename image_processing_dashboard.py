import streamlit as st
import numpy as np
import cv2
from PIL import Image

# Histagram matching function

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


else:
    st.warning("Please upload an image.")