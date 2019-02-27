"""Final Project -- Stereo Correspondence"""
import numpy as np
import cv2
import os
import math
import maxflow
import stereo
from scipy import stats

def part_1a():
    # load two files
    image1_filename = "input/noisy.png"
    image2_filename = "input/noisy2.png"
    
    image1 = cv2.imread(image1_filename, cv2.IMREAD_GRAYSCALE)
    image2 = cv2.imread(image2_filename, cv2.IMREAD_GRAYSCALE)
    
    disp3 = final.sumOfSquaredDifferences(image1, image2, 3)
    disp5 = final.sumOfSquaredDifferences(image1, image2, 5)
    disp7 = final.sumOfSquaredDifferences(image1, image2, 7)
    
    disp3 = final.norm_scale_color(disp3)
    disp5 = final.norm_scale_color(disp5)
    disp7 = final.norm_scale_color(disp7)
    
    cv2.imwrite("blah/disparity1a3.png", disp3)
    cv2.imwrite("blah/disparity1a5.png", disp5)
    cv2.imwrite("blah/disparity1a7.png", disp7)
    
def part_1b():
    # load two files
    image1_filename = "input/Piano-perfect/im0.png"
    image2_filename = "input/Piano-perfect/im1.png"
    
    image1 = cv2.imread(image1_filename, cv2.IMREAD_GRAYSCALE)
    image2 = cv2.imread(image2_filename, cv2.IMREAD_GRAYSCALE)
    image1 = cv2.resize(image1, (0,0), fx=0.25, fy=0.25)
    image2 = cv2.resize(image2, (0,0), fx=0.25, fy=0.25)
    
    disp3 = final.sumOfSquaredDifferences(image1, image2, 3)
    disp5 = final.sumOfSquaredDifferences(image1, image2, 5)
    disp7 = final.sumOfSquaredDifferences(image1, image2, 7)
    
    disp3 = final.norm_scale_color(disp3)
    disp5 = final.norm_scale_color(disp5)
    disp7 = final.norm_scale_color(disp7)
    
    cv2.imwrite("output/disparity1b3.png", disp3)
    cv2.imwrite("output/disparity1b5.png", disp5)
    cv2.imwrite("output/disparity1b7.png", disp7)
	
def part_2a():
    # load two files
    image1_filename = "input/noisy.png"
    image2_filename = "input/noisy2.png"
    
    image1 = cv2.imread(image1_filename, cv2.IMREAD_GRAYSCALE)
    image2 = cv2.imread(image2_filename, cv2.IMREAD_GRAYSCALE)
    
    disp = final.kolmogorov(image1, image2)
    disp = final.handle_infinities_scale_color(disp)
    
    cv2.imwrite("output/disparity2a.png", disp)
    
def part_2b():
    # load two files
    image1_filename = "input/Flowers-perfect/im0.png"
    image2_filename = "input/Flowers-perfect/im1.png"
    
    image1 = cv2.imread(image1_filename, cv2.IMREAD_GRAYSCALE)
    image2 = cv2.imread(image2_filename, cv2.IMREAD_GRAYSCALE)
    image1 = cv2.resize(image1, (0,0), fx=0.25, fy=0.25)
    image2 = cv2.resize(image2, (0,0), fx=0.25, fy=0.25)
    
    disp = final.kolmogorov(image1, image2)
    disp = final.handle_infinities_scale_color(disp)
    
    cv2.imwrite("output/disparity2b.png", disp)
	
if __name__ == "__main__":
    part_1a()
    #part_1b()
    #part_2a()
    #part_2b()
    
