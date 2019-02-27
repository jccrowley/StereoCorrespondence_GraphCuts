"""Final Project -- Stereo Correspondence"""
import numpy as np
import cv2
import os
import math
import maxflow
import sys
from scipy import stats

VAR_ALPHA = 0
VAR_ABSENT = 1
OCCLUDED = np.inf
COMPUTE_SMOOTHNESS = True
COMPUTE_DATA_OCC = True

def sumOfSquaredDifferences(image1, image2, w):
    
    disp = np.zeros(image1.shape)
    
    for y in range(w, len(image1)-w):
        for x in range(w, len(image1[y])-w):
            res = cv2.matchTemplate(image1[y-w:y+w,:],image2[y-w:y+w,x-w:x+w],cv2.cv.CV_TM_SQDIFF) 
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            disp[y,x] = abs(x-min_loc[0])
            
    return disp
    
# Utility function
def normalize_and_scale(image_in, scale_range=(0, 255)):
    image_out = np.zeros(image_in.shape)
    cv2.normalize(image_in, image_out, alpha=scale_range[0],
                  beta=scale_range[1], norm_type=cv2.NORM_MINMAX)

    return image_out

def handle_infinities(image_in):
    flat = np.unique(image_in.flatten())
    flat.sort()
    large = flat[-2]
    image_out = np.where(image_in == np.inf, large+1, image_in)
    flat = image_out.flatten()
    flat = flat.astype(np.uint8)
    #print stats.describe(image_in)
    print np.bincount(flat)

def handle_infinities_scale_color(image_in):
    locations = np.where(image_in == np.inf, 0, 1)
    image_out = np.where(image_in == np.inf, 0, image_in)
    average = sum(image_out) / np.sum(locations)
    image_out = np.where(locations == 0, average, image_in)
    disp = normalize_and_scale(image_out)
    disp = disp.astype(np.uint8)
    image_out = cv2.applyColorMap(disp, cv2.COLORMAP_JET)
    for i in range(3):
        image_out[:,:,i] = np.where(locations == 0, 0, image_out[:,:,i])
    return image_out

def norm_scale_color(disp):
    disp = normalize_and_scale(disp)
    disp = disp.astype(np.uint8)
    return cv2.applyColorMap(disp, cv2.COLORMAP_JET)
							
def kolmogorov(image1, image2, K=None):
    image1 = image1.astype(np.float32)
    image2 = image2.astype(np.float32)
    disp = range(1, 10)
    dissim = absolute_dissimilarity
    alphaRange = 15
    iters = 4
    if K == None:
        K = compute_k(alphaRange, image1, image2, dissim)
    print K
    oldE = sys.maxsize
    lambd = K/5
    dispL = np.ones(image1.shape) * OCCLUDED
    
    for ite in range(iters):
        alphas = np.random.permutation(alphaRange)
        #alphas = np.arange(alphaRange)
        #alphas = [0]
        #alphas = [0, 5]
        for alpha in alphas:

            g = maxflow.Graph[float](2, 2)
            varsO = []
            varsOnodes = []
            varsA = []
            varsAnodes = []

            # build varsO, varsA, and graph nodes
            for r in range(dispL.shape[0]):
                for p in range(dispL.shape[1]):
                    if dispL[r, p] == OCCLUDED: # occluded
                        varsO.append(VAR_ABSENT)
                        varsOnodes.append(-1)
                        if(p + alpha < image1.shape[1]): # assignment legal
                            varsA.append({"pixel" : (r, p), "disp" : alpha})
                            varsAnodes.append(g.add_nodes(1))
                        else:
                            varsA.append(VAR_ABSENT)
                            varsAnodes.append(-1)
                    else: # not occluded
                        if(dispL[r, p] != alpha):
                            varsO.append({"pixel" : (r, p), "disp" : dispL[r, p]})
                            varsOnodes.append(g.add_nodes(1))
                            if(p + alpha < image1.shape[1]): # assignment legal
                                varsA.append({"pixel" : (r, p), "disp" : alpha})
                                varsAnodes.append(g.add_nodes(1))
                            else:
                                varsA.append(VAR_ABSENT)
                                varsAnodes.append(-1)
                        else:
                            varsA.append(VAR_ALPHA)
                            varsO.append(VAR_ALPHA)
                            varsAnodes.append(-1)
                            varsOnodes.append(-1)

            # add edges

            # data/occlusion terms
            active_penalty = 0.
            if COMPUTE_DATA_OCC:
                for r in range(dispL.shape[0]):
                    for c in range(dispL.shape[1]):
                        n = (dispL.shape[1] * r) + c
                        if is_var(varsA[n]):
                            p = varsA[n]["pixel"]
                            a = varsA[n]["disp"]
                            D = dissim(image1, image2, p, (p[0],int(p[1]+a))) - K
                            g.add_tedge(varsAnodes[n], D, 0)
                        if varsA[n] == VAR_ALPHA:
                            p = (r, c)
                            a = alpha
                            D = dissim(image1, image2, p, (p[0],int(p[1]+a))) - K
                            active_penalty += D
                        if is_var(varsO[n]):
                            p = varsO[n]["pixel"]
                            a = varsO[n]["disp"]
                            D = dissim(image1, image2, p, (p[0],int(p[1]+a))) - K
                            g.add_tedge(varsOnodes[n], 0, D)

            # smoothwise/uniqueness terms
            for r in range(dispL.shape[0]):
                for c in range(dispL.shape[1]):
                    i1 = (dispL.shape[1] * r) + c
                    
                    # uniqueness
                    if is_var(varsO[i1]) and is_var(varsA[i1]):
                        forbid01(g, varsOnodes[i1], varsAnodes[i1])
                        ia = int(varsO[i1]["disp"])
                        c2 = c+ia-alpha
                        i2 = (dispL.shape[1] * r) + c2
                        if c2 > dispL.shape[1]:
                            print "OTHER DISASTER"
                        forbid01(g, varsOnodes[i1], varsAnodes[i2])

                    # smoothness
                    if COMPUTE_SMOOTHNESS:
                        indices = (c, r)
                        for i in range(2):
                            if indices[i] != dispL.shape[1-i]-1:
                                r2 = r+i
                                c2 = c+(1-i)
                                if c2+alpha < dispL.shape[1]:
                                    i2 = (dispL.shape[1] * r2) + c2
                                    pen = lambd
                                    d1 = int(image1[r, c]) - int(image1[r2, c2])
                                    d2 = int(image2[r, c+alpha]) - int(image2[r2, c2+alpha])
                                    if max(d1, d2) < 8:
                                        pen *= 3.

                                    # pairwise varsA
                                    if is_var(varsA[i1]):
                                        if is_var(varsA[i2]):
                                            pairwise_term (g, varsAnodes[i1], varsAnodes[i2], 0, pen, pen, 0) # add term

                                    # pairwise varsO
                                    if is_var(varsO[i1]):
                                        if is_var(varsO[i2]):
                                            pairwise_term (g, varsAnodes[i1], varsAnodes[i2], 0, pen, pen, 0) # add term

                                    # unary varsA
                                    if is_var(varsA[i1]) and varsA[i2] == VAR_ALPHA:
                                        g.add_tedge(varsAnodes[i1], 0, pen)
                                    if is_var(varsA[i2]) and varsA[i1] == VAR_ALPHA:
                                        g.add_tedge(varsAnodes[i2], 0, pen)

                                    # unary varsO
                                    if is_var(varsO[i1]) and not is_var(varsO[i2]):
                                        g.add_tedge(varsOnodes[i1], 0, pen)
                                    if is_var(varsO[i2]) and not is_var(varsO[i1]):
                                        g.add_tedge(varsOnodes[i2], 0, pen)

            # all terms encoded, can now compute cut
            flow = g.maxflow()
            E = flow + active_penalty

            # update disparity map
            num_Active = 0
            if E < oldE:
                oldE = E
                for r in range(dispL.shape[0]):
                    for c in range(dispL.shape[1]):
                        i = (dispL.shape[1] * r) + c
                        
                        if varsOnodes[i] != -1 and g.get_segment(varsOnodes[i]) == 0 and varsAnodes[i] != -1 and g.get_segment(varsAnodes[i]) == 1:
                            print "DISASTER"
                            print g.get_segment(varsOnodes[i]), g.get_segment(varsAnodes[i])
                            print varsOnodes[i], varsAnodes[i]

                        if varsOnodes[i] != -1 and g.get_segment(varsOnodes[i]) == 1:
                            dispL[r, c] = OCCLUDED

                        if varsAnodes[i] != -1 and g.get_segment(varsAnodes[i]) == 1:
                            dispL[r, c] = alpha
                            num_Active += 1
                            
                # disparity map updated -- check energy equals
                #print alpha
                #print E, flow, active_penalty
                #print compute_energy(dispL, image1, image2, K), compute_smoothness(dispL, image1, image2, K), compute_data_occ(dispL, image1, image2, K)
                #print num_Active
            		
    return dispL
				
				
def pairwise_term(g, n1, n2, A, B, C, D):
	g.add_tedge(n1, D, B)
	g.add_tedge(n2, 0, A-B)
	g.add_edge(n1, n2, 0, B+C-A-D)
	
def forbid01(g, n1, n2):
    g.add_edge(n1, n2, sys.maxsize, 0)
	
def is_var(v):
	return v != VAR_ALPHA and v != VAR_ABSENT

def compute_k(alphaRange, image1, image2, dissim):
    K = int((alphaRange+2)/4)
    s = 0
    i = 0
    for r in range(alphaRange, image1.shape[0]-alphaRange):
        for c in range(alphaRange, image2.shape[1]-alphaRange):
            i += 1
            Das = []
            for alpha in range(alphaRange):
                #Das.append(abs(int(image1[r, c])-int(image2[r,c+alpha])))
                Das.append(dissim(image1, image2, (r, c), (r, c+alpha)))
            Das.sort()
            s += Das[K]
    return int(s/i)

def compute_energy(disp, image1, image2, K):
    return compute_smoothness(disp, image1, image2, K) + compute_data_occ(disp, image1, image2, K)
    
def compute_smoothness(disp, image1, image2, K):
     # compute smoothness term
    lambd = int(K/5) 
    smoothness = 0
    pen1 = pen2 = 0
    for r in range(disp.shape[0]):
        for c in range(disp.shape[1]):
            i1 = (disp.shape[1] * r) + c
            alpha1 = disp[r, c]

            indices = (c, r)
            for i in range(2):
                if indices[i] != disp.shape[1-i]-1:
                    r2 = int(r+i)
                    c2 = int(c+(1-i))
                    alpha2 = disp[r2, c2]
                    
                    if alpha1 != alpha2:
                    
                        if alpha1 != np.inf and c2+alpha1>disp.shape[1]:
                            pen1 = lambd
                            d1 = int(image1[r, c]) - int(image1[r2, c2])
                            d2 = int(image2[r, int(c+alpha1)]) - int(image2[r2, int(c2+alpha1)])
                            if max(d1, d2) < 8:
                                pen1 *= 3
                        
                        if alpha2 != np.inf:
                            pen2 = lambd
                            d1 = int(image1[r, c]) - int(image1[r2, c2])
                            d2 = int(image2[r, int(c+alpha2)]) - int(image2[r2, int(c2+alpha2)])
                            if max(d1, d2) < 8:
                                pen2 *= 3

                        smoothness += pen1 + pen2
    return smoothness

def compute_data_occ(disp, image1, image2, K):
    data_occ = 0
    num_active = 0
    for r in range(disp.shape[0]):
        for c in range(disp.shape[1]):
            if disp[r, c] != np.inf:
                p = (r, c)
                a = disp[r, c]
                
                D = abs(int(image1[p])-int(image2[p[0],int(p[1]+a)])) - K
                #print D
                data_occ += D
                num_active += 1
                 
    #print "num active from method: " + str(num_active)
    return data_occ

def absolute_dissimilarity(image1, image2, p1, p2):
    return abs(image1[p1]-image2[p2])

def squared_dissimilarity(image1, image2, p1, p2):
    return (image1[p1]-image2[p2])**2
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
                             