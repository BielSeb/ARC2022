import cv2 as cv
import numpy as np
from os import listdir
from os.path import isfile, join

IMAGES_PATH = 'C:\\Users\\Sebastian Bielmeier\\Documents\\GitHub\\Projekt2_try_stitching_3\\src'
OUT_PATH = 'C:\\Users\\Sebastian Bielmeier\\Documents\\GitHub\\Projekt2_try_stitching_3\\out'

def main():
    image_paths = sorted([f for f in listdir(IMAGES_PATH) if isfile(join(IMAGES_PATH, f)) and f.endswith('jpg')])
    imgs = list()
    for img_path in image_paths:
        if isfile(join(IMAGES_PATH, img_path)):
            imgs.append(cv.imread(join(IMAGES_PATH, img_path), cv.IMREAD_GRAYSCALE))
    # ---> choose sample here [0,1,2] <---
    sample = 0
    # transform images onto a plane
    im1 = remappImage(imgs[sample*4])
    w1 = im1.shape[1]
    im1_left = im1[:,:int(w1/2)]
    im1_right = im1[:,-int(w1/2):]
    im2 = remappImage(imgs[sample*4 + 1])
    w2 = im2.shape[1]
    im2_left = im2[:,:int(w2/2)]
    im2_right = im2[:,-int(w2/2):]
    im3 = remappImage(imgs[sample*4 + 2])
    w3= im3.shape[1]
    im3_left = im3[:,:int(w3/2)]
    im3_right = im3[:,-int(w3/2):]
    im4 = remappImage(imgs[sample*4 + 3])
    w4 = im4.shape[1]
    im4_left = im4[:,:int(w4/2)]
    im4_right = im4[:,-int(w4/2):]
    # stitch images 1 and 2
    kps1, des1 = orbDetection(im1_right)
    kps2, des2 = orbDetection(im2_left)
    matches = bfMatcher(des1, des2)
    pts = computeMatchVectors(matches, kps1, kps2)
    new_img1 = stitch(im1_right, im2_left, pts)
    # determine offset for appending first two stitches
    if pts[1]<0:
        offset1 = pts[1]
    else:
        offset1 = 0
    # stitch images 2 and 3
    kps1, des1 = orbDetection(im2_right)
    kps2, des2 = orbDetection(im3_left)
    matches = bfMatcher(des1, des2)
    pts = computeMatchVectors(matches, kps1, kps2)
    new_img2 = stitch(im2_right, im3_left,pts)
    # append first two stitches
    new_img12 = stitch(new_img1, new_img2, [new_img1.shape[1], offset1])
    # stitch images 3 and 4
    kps1, des1 = orbDetection(im3_right)
    kps2, des2 = orbDetection(im4_left)
    matches = bfMatcher(des1, des2)
    pts = computeMatchVectors(matches, kps1, kps2)
    new_img3 = stitch(im3_right, im4_left, pts)
    # determine offset for appending last two stitches
    if pts[1]<0:
        offset2 = pts[1]
    else:
        offset2 = 0
    # stitch images 4 and 1
    kps1, des1 = orbDetection(im4_right)
    kps2, des2 = orbDetection(im1_left)
    matches = bfMatcher(des1, des2)
    pts = computeMatchVectors(matches, kps1, kps2)
    new_img4 = stitch(im4_right, im1_left, pts)
    #append last two stitches 
    new_img34 = stitch(new_img3, new_img4, [new_img3.shape[1], offset2])
    # append all together
    offset_left = np.where(new_img12[:,-1]>0)[0][0]
    offset_right = np.where(new_img34[:,0]>0)[0][0]
    new_img = stitch(new_img12, new_img34, [new_img12.shape[1], offset_right-offset_left])
    # save stitched image
    cv.imwrite('out/sample' + str(sample) + '.jpg', new_img)
        
    cv.imshow('stitched image', new_img)
    cv.waitKey()
    cv.destroyAllWindows()
    
def stitch(img1, img2, vec):
    # determine transformation from vector t = [[tx],[ty]]
    tx, ty = int(vec[0]), int(vec[1])
    # size of input images
    y2, x2 = img2.shape[:2]
    y1, x1 = img1.shape[:2]
    # shape of new image 
    new_rows = max(y1, y2) * 2
    new_cols = max(x1, x2) * 2
    # assign new image matrix with target size
    stitched = np.zeros((new_rows,new_cols), np.uint8)
    temp = stitched.copy()
    if (tx<0):
        if (ty<0):
            temp[-ty:y2-ty, -tx:x2-tx] = img2 
            stitched[0:y1, 0:x1] = img1
            stitched = cv.addWeighted(temp,.8,stitched,.8,0)
        else:
            temp[0:y2, -tx:x2-tx] = img2 
            stitched[ty:y1+ty, 0:x1] = img1
            stitched = cv.addWeighted(temp,.8,stitched,.8,0)
    else:
        if (ty<0):
            temp[-ty:y2-ty, tx:x2+tx] = img2 
            stitched[0:y1, 0:x1] = img1
            stitched = cv.addWeighted(temp,.8,stitched,.8,0)
        else:
            temp[0:y2, tx:x2+tx] = img2 
            stitched[ty:y1+ty, 0:x1] = img1
            stitched = cv.addWeighted(temp,.8,stitched,.8,0)
    x, y = np.nonzero(stitched)
    xl,xr = x.min(),x.max()
    yl,yr = y.min(),y.max()
    # do not return empty area empty areas
    cropped = stitched[xl:xr+1, yl:yr+1]
    return cropped

def computeMatchVectors(matches, kps1, kps2):
    MatchVec = np.zeros(2,np.float64)
    # get empty arrays form matched points
    pts1 = np.zeros(2*len(matches), np.int32)
    pts1 = pts1.reshape((-1,2))
    pts2 = pts1.copy()
    # copy matched points to empty arrays
    for i in range(0, len(matches)):
        pts1[i,:] = kps1[matches[i].queryIdx].pt
        pts2[i,:] = kps2[matches[i].trainIdx].pt
    # get vectors from key points
    vectors = pts2-pts1
    # compute histogram
    n, bins= np.histogram(vectors[:,0], bins=25)
    # get hightest bin
    elem = np.argmax(n)
    try:
        # determine weighted mean of 3 bins around max of histogram
        MatchVec[0] = np.average([((bins[elem-1]+bins[elem])/2),
                                ((bins[elem]+bins[elem+1])/2),
                                ((bins[elem+1]+bins[elem+2])/2)], axis=0, weights=n[elem-1:elem+2])
    except:
        MatchVec[0] = (bins[elem]+bins[elem+1])/2

    # compute histogram
    n, bins= np.histogram(vectors[:,1], bins=25)
    # get highest bin
    elem = np.argmax(n)
    try:
        # determine weighted mean of 3 bins around max of histogram
        MatchVec[1] = np.average([((bins[elem-1]+bins[elem])/2),
                                ((bins[elem]+bins[elem+1])/2),
                                ((bins[elem+1]+bins[elem+2])/2)], axis=0, weights=n[elem-1:elem+2])
    except:
        MatchVec[1] = (bins[elem]+bins[elem+1])/2
    return MatchVec

def max(a, b):
    if a>=b:
        return a
    else:
        return b
        
def orbDetection (img):
    orb = cv.ORB_create()
    kps, des = orb.detectAndCompute(img, None)
    return kps, des    

def bfMatcher(des1, des2):
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    # Match descriptors.
    matches = bf.match(des1,des2)
    # Sort them in the order of their distance.
    matches = sorted(matches, key = lambda x:x.distance)
    return matches

def remappImage(img):
    # get image size
    height, width = img.shape[:2]
    # assign empty mapping matrices
    radius = int(width/2)
    map_y = np.zeros((height, int(width*np.pi/2)), dtype=np.float32)
    map_x = map_y.copy()
    # compute and assign column mapping matrix
    i = np.array(range(map_x.shape[1]))
    x_val = radius * -np.cos(i/radius) + radius   
    map_x[:, :] = np.tile(x_val,(map_y.shape[0],1))
    # assign row mapping matrix
    y = range(map_x.shape[0])
    map_y[:, :] = np.tile(y,(map_x.shape[1],1)).transpose()
    # apply remapping function        
    remapped = cv.remap(img, map_x, map_y, cv.INTER_LINEAR)
    newwidth = remapped.shape[1]
    return remapped[:, int(0.07*newwidth):int(0.93*newwidth)]

if __name__ == "__main__":
    main()


