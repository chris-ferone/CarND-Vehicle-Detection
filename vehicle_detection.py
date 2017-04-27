# Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images
# train a classifier Linear SVM classifier
# normalize your features and randomize a selection for training and testing
# Implement a sliding-window technique and use your trained classifier to search for vehicles in images

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pickle
import cv2
from lesson_functions import *
# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from heat_map import *

dist_pickle = pickle.load( open("svc_pickle.p", "rb" ) )
svc = dist_pickle["svc"]
X_scaler = dist_pickle["X_scaler"]
orient = dist_pickle["orient"]
pix_per_cell = dist_pickle["pix_per_cell"]
cell_per_block = dist_pickle["cell_per_block"]

# Define a single function that can extract features using hog sub-sampling and make predictions
def find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size = (32, 32), hist_bins = 32):
    draw_img = np.copy(img)
    img = img.astype(np.float32) / 255
    box_list = []
    img_tosearch = img[ystart:ystop, :, :]
    ctrans_tosearch = convert_color(img_tosearch, conv='RGB2YUV')
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1] / scale), np.int(imshape[0] / scale)))

    ch1 = ctrans_tosearch[:, :, 0]
    ch2 = ctrans_tosearch[:, :, 1]
    ch3 = ctrans_tosearch[:, :, 2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell) - 1
    nyblocks = (ch1.shape[0] // pix_per_cell) - 1
    nfeat_per_block = orient * cell_per_block ** 2
    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell) - 1
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step

    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)

    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb * cells_per_step
            xpos = xb * cells_per_step
            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_feat2 = hog2[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_feat3 = hog3[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

            xleft = xpos * pix_per_cell
            ytop = ypos * pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop + window, xleft:xleft + window], (64, 64))

            # Get color features
            spatial_features = bin_spatial(subimg, size=spatial_size)
            hist_features = color_hist(subimg, nbins=hist_bins)

            # Scale features and make a prediction
            test_features = X_scaler.transform(np.hstack((hog_features, spatial_features, hist_features)).reshape(1, -1))
            test_prediction = svc.predict(test_features)

            if test_prediction == 1:
                xbox_left = np.int(xleft * scale)
                ytop_draw = np.int(ytop * scale)
                win_draw = np.int(window * scale)
                x1 = xbox_left
                y1 = ytop_draw + ystart
                x2 = xbox_left + win_draw
                y2 = ytop_draw + win_draw + ystart
                cv2.rectangle(draw_img, (x1, y1), (x2, y2), (0, 0, 255), 6)
                box_list.append(((x1, y1), (x2, y2)))

    return draw_img, box_list

def pipeline(img):
    ystart_array = [400, 400, 400]
    ystop_array =  [800, 600, 500]
    scale_array =  [2,   1.5, 1.2]
    #scale should increase towards bottom of image y => 720 because images are larger, and scale should decrease at top of image y => 0

    combined_boxlist = []
    for i in range(0, len(ystart_array)):
        [out_img, box_list] = find_cars(img, ystart_array[i], ystop_array[i], scale_array[i], svc, X_scaler, orient, pix_per_cell, cell_per_block)
        #plt.imshow(out_img)
        #plt.show()
        combined_boxlist.extend(box_list)

    if len(combined_boxlist) > 0:
        heat = np.zeros_like(img[:, :, 0]).astype(np.float)
        # Add heat to each box in box list
        heat = add_heat(heat, combined_boxlist)

        # Apply threshold to help remove false positives
        heat = apply_threshold(heat, 1)

        # Visualize the heatmap when displaying
        heatmap = np.clip(heat, 0, 255)

        # Find final boxes from heatmap using label function
        labels = label(heatmap)
        #print(labels[1], 'cars found')
        draw_img = draw_labeled_bboxes(np.copy(img), labels)
        #print("labels: ", labels[1])


        fig = plt.figure()
        #ax1 = fig.add_subplot(111)
        #a1=ax1.imshow(draw_img)
        # #fig.colorbar(a1)
        # ax1.set_title('Final Car Positions')
        # ax2 = fig.add_subplot(312)
        # ax2.imshow(heatmap, cmap='hot')
        # ax2.set_title('Heat Map')
        ax3 = fig.add_subplot(111)
        box_img = draw_boxes2(np.copy(img), combined_boxlist)
        ax3.imshow(box_img, cmap='hot')
        # ax3.set_title('Original')
        fig.tight_layout()
    else:
        draw_img = img
        #print("no vehicles detected")


    return draw_img



UseStillImage = True
box_list = []

if UseStillImage:

    img = mpimg.imread('frames/frame934.jpg')
    #out_img = find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)
    out_img=pipeline(img)
    fig = plt.figure()
    plt.imshow(out_img)



    plt.show()

else:
    input_clip = VideoFileClip('project_video.mp4')#.subclip(38,42)
    output_clip = input_clip.fl_image(pipeline)
    output_clip.write_videofile('output.mp4', audio=False)