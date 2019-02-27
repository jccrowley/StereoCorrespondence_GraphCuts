"""Final Project -- Stereo Correspondence"""
import numpy as np
import cv2
import os
import math
import maxflow
import sys
from scipy import stats
import loadpfm
import stereo

def percent_correct(gt, disp):
    print np.unique(gt.flatten())
    print np.unique(disp.flatten())
    print gt - gt
    print disp - disp
    diff = gt - disp
    diff = np.abs(diff)
    diff = np.where(diff == 0, 1, diff)
    diff = np.where(diff != 1, 0, 1)
    return 100. * (np.sum(diff) / diff.size)

def average_difference(gt, disp):
    return np.average(np.abs(gt - disp))

filename = "input/Flowers-perfect/disp1.pfm"
ground_truth_flowers, _ = loadpfm.load_pfm(open(filename, "rb"))
ground_truth_flowers = ground_truth_flowers.T
ground_truth_flowers = np.rot90(ground_truth_flowers)
ground_truth_flowers = cv2.resize(ground_truth_flowers, (0,0), fx=0.25, fy=0.25)
ground_truth_flowers *= 0.25

image1_filename = "input/Flowers-perfect/im0.png"
image2_filename = "input/Flowers-perfect/im1.png"
image1 = cv2.imread(image1_filename, cv2.IMREAD_GRAYSCALE)
image2 = cv2.imread(image2_filename, cv2.IMREAD_GRAYSCALE)
image1 = cv2.resize(image1, (0,0), fx=0.25, fy=0.25)
image2 = cv2.resize(image2, (0,0), fx=0.25, fy=0.25)

disp3 = final.sumOfSquaredDifferences(image1, image2, 3)
"""disp5 = final.sumOfSquaredDifferences(image1, image2, 5)
disp7 = final.sumOfSquaredDifferences(image1, image2, 7)"""

"""print percent_correct(ground_truth_flowers, disp3)
print percent_correct(ground_truth_flowers, disp5)
print percent_correct(ground_truth_flowers, disp7)
print average_difference(ground_truth_flowers, disp3)
print average_difference(ground_truth_flowers, disp5)
print average_difference(ground_truth_flowers, disp7)"""

disp = final.kolmogorov(image1, image2)
print percent_correct(ground_truth_flowers, disp)
#print average_difference(ground_truth_flowers, disp)"""


#ground_truth_flowers = final.handle_infinities_scale_color(ground_truth_flowers)
#cv2.imwrite("output/ground-truth-flowers.png", ground_truth_flowers)
