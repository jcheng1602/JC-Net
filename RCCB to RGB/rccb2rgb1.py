#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from scipy.misc import imread, imsave, imresize
import cv2

import matplotlib.pyplot as plt
from scipy import interpolate


def get_r1_r2(img_raw_companded):
    """
    get remapped IRBI

    Reference: https://sourcecode.socialcoding.bosch.com/users/let1lr/repos/imageaccesslibrary/browse/doc/howToDecompand/howto_decompand.pdf
    """
    try:
        # receive the ratio indices and dcg settings from the header
        idx_12 = (img_raw_companded[6][268] >> 6) & 0x3
        idx_23 = (img_raw_companded[6][268] >> 8) & 0x3
        dcg    = (img_raw_companded[6][520] >> 6) & 0x1
        print ("ok")
        if 1:
            print("r12 = {}, r_23 = {}, dcg = {} (0 = lg, 1 = hg)".format(4*2**idx_12, 4*2**idx_23, dcg))
    except:
        raise ValueError(
            'Invalid images for LUV conversion! Check if your images have header and trailers and try again.')
    # calculate the LUT id [0,17]
    idx_LUT = idx_23 + 3 * (idx_12 + 3 * dcg)
    if (idx_LUT > 17):
        idx_LUT = 17
    # apply tone-mapping/remapping
    # in GEN3 the interpolated I channel of the 12-bit-remapped image is treated as the luminance component
    # L = np.take(LUT_remapping[idx_LUT], img_raw_companded)
    # get ratio R1 and R2
    R1 = 1 << (idx_12 + 2)
    R2 = 1 << (idx_23 + 2)
    # delete header and trailer lines, np.ndarray.astype(L[8:-8], dtype='uint16'),
    return R1, R2

def _get_P(r1, r2):

    C0=64
    P0=0

    if (r1==4):
        C1=2048
        P2=2**14
        C2=2944
        P1=2048
        if (r2==4):
            P3=2**16
            C3=3712

        elif(r2==8):
            P3=2**17
            C3=3840
            
        else:
            P3=2**18
            C3=3904
            
    elif (r1==8):
        C1=2048
        P2=2**15
        C2=3008
        P1=2048
        if (r2==4):
            P3=2**17
            C3=3776

        elif(r2==8):
            P3=2**18
            C3=3904
            
        else:
            P3=2**19
            C3=3968 
            
    elif (r1==16):
        C1=2048
        P1=2048
        P2=2**16
        C2=3040
        if (r2==4):
            P3=2**18
            C3=3808

        elif(r2==8):
            P3=2**19
            C3=3936
            
        else:
            P3=2**20
            C3=4000

    return P0, P1, P2, P3, C0, C1, C2, C3

def _get_20bit_from_12bit(P0,P1,P2,P3,C0,C1,C2,C3,img_raw_companded):

    print(P0,P1,P2,P3,C0,C1,C2,C3)
    f_linear = interpolate.interp1d(np.array([C0,C1,C2,C3],dtype=np.uint32),np.array([P0,P1,P2,P3],dtype=np.uint32),
        bounds_error=False,fill_value=(P0,P3))
    img_raw_decompand = f_linear(img_raw_companded).astype(np.uint32)

    return img_raw_decompand

def _get_c_expand(img_raw_companded):

    C_expand = img_raw_companded.copy()
    height,width = C_expand.shape

    for i in range(1,height,2):
        for j in range(0,width,2):
            near_id = [[i,j-1],[i,j+1],[i-1,j],[i+1,j]]
            valid_id = [[h,w] for h ,w in near_id if h>=0 and h <height and w >=0 and w< width]

            valid_id = [[h for h,w in valid_id],[w for h,w in valid_id]]
            mean_value = np.mean(C_expand[valid_id]).astype(np.uint32)

            C_expand[i,j] = mean_value

    for i in range(0,height,2):
        for j in range(1,width,2):
            near_id = [[i,j-1],[i,j+1],[i-1,j],[i+1,j]]
            valid_id = [[h,w] for h ,w in near_id if h>=0 and h <height and w >=0 and w< width]

            valid_id = [[h for h,w in valid_id],[w for h,w in valid_id]]
            mean_value = np.mean(C_expand[valid_id]).astype(np.uint32)

            C_expand[i,j] = mean_value

            # print valid_id
            # print mean_value
            # break

    return C_expand

def get_img_decompand(img_raw_companded, r1=4, r2=8):

    img_raw_companded = img_raw_companded.astype(np.uint32)
    P0, P1, P2, P3, C0, C1, C2, C3 = _get_P(r1,r2)
    img_raw_decompand = _get_20bit_from_12bit(P0, P1, P2, P3, C0, C1, C2, C3, img_raw_companded)

    img_raw_decompand.astype(np.uint32)

    return img_raw_decompand


def get_img_crb(img_raw_decompand):
    """
    """
    img_raw_decompand = img_raw_decompand.astype(np.uint16)  
    hight, width = img_raw_decompand.shape

    B_raw = img_raw_decompand[1::2, ::2]
    R_raw = img_raw_decompand[::2, 1::2]

 
    B_expand = cv2.resize(B_raw,(width,hight),interpolation=cv2.INTER_LINEAR)
    R_expand = cv2.resize(R_raw,(width,hight),interpolation=cv2.INTER_LINEAR)

    C_expand = _get_c_expand(img_raw_decompand)

    C_expand = np.expand_dims(C_expand,2)
    B_expand = np.expand_dims(B_expand,2)
    R_expand = np.expand_dims(R_expand,2)

    #print('B_raw',B_raw.shape, B_raw.max(),B_raw.min())
    #print('B_expand',B_expand.shape, B_expand.max(),B_expand.min())

    crb = np.concatenate((C_expand,R_expand,B_expand),axis=2)

    return crb

def crb2rgb(crb_expand, r1=4, r2=8):

    A_matrix = np.array([[0.2415,-0.0375,-0.4],
                        [0.2428,-0.1709,-0.2620],
                        [-0.0527,-0.0320,0.4264]]) * 0.016/r1/r2

    rgb = (np.dot(A_matrix,crb_expand.reshape((-1,3)).T).T).reshape(crb_expand.shape)

    # print('rgb',rgb.shape,rgb.max(),rgb.min())

    return rgb

def do_rgb_clamp(rgb):

    rgb = rgb.copy()
    height, width = rgb.shape[:2]

    for m in range(height):
        for n in range(width):
            temp_rgb = rgb[m,n,:]

            if np.any(temp_rgb<0.) or np.any(temp_rgb>1.):
                # print('enter')
                lamda = np.mean(temp_rgb)
                if lamda <= 0.:
                    rgb[m,n,:] = [0.,0.,0.]
                elif lamda >=1.:
                    rgb[m,n,:] = [1.,1.,1.]
                else:
                    mu_max = 0. 

                    r,g,b = temp_rgb
                    if r<0.:
                        mu_max = max(mu_max, 3.*r/(2.*r-g-b))
                    elif r>1.:
                        mu_max = max(mu_max, (3.*r-3.)/(2.*r-g-b))

                    if g<0.:
                        mu_max = max(mu_max, 3.*g/(2.*g-r-b))
                    elif g>1.:
                        mu_max = max(mu_max, (3.*g-3)/(2.*g-r-b))

                    if b<0.:
                        mu_max = max(mu_max, 3.*b/(2.*b-r-g))
                    elif b>1.:
                        mu_max = max(mu_max, (3.*b-3.)/(2.*b-r-g))

                    rgb[m,n,0] = (1. - mu_max)*r + mu_max*lamda
                    rgb[m,n,1] = (1. - mu_max)*g + mu_max*lamda
                    rgb[m,n,2] = (1. - mu_max)*b + mu_max*lamda

    return rgb


def white_balance(rgb):

    rgb = rgb.copy()
    r,g,b = np.split(rgb,3,axis=2)

    r_mean = r.mean()
    g_mean = g.mean()
    b_mean = b.mean()

    kr = (r_mean+g_mean+b_mean)/r_mean/3.
    kg = (r_mean+g_mean+b_mean)/g_mean/3.
    kb = (r_mean+g_mean+b_mean)/b_mean/3.

    r = r*kr
    g = g*kg 
    b = b*kb 

    new_im = np.concatenate((r,g,b),axis=2)

    return new_im

if __name__ == "__main__":

    img_path = '/home/riki/Desktop/RCCB/LB-XL_8377_20150820_131902_frame561.png'

    I_raw = imread(img_path)
    #print('I_raw',I_raw.shape,I_raw.max(),I_raw.min())
    img_raw = np.ndarray.astype(I_raw, dtype='uint16')
    #print('img_raw',img_raw.shape,img_raw.max(),img_raw.min())
    r1, r2 = get_r1_r2(img_raw) 
   # print (r1,r2)

  
    img_raw = img_raw[8:-8,:]
   # print('img_raw',img_raw.shape,img_raw.max(),img_raw.min())

    img_raw_decompand = get_img_decompand(img_raw,r1,r2)
    #print('img_raw_decompand',img_raw_decompand.shape,img_raw_decompand.max(),img_raw_decompand.min())

    crb = get_img_crb(img_raw_decompand)
    #print('crb', crb.shape,crb.max(), crb.min())

    rgb = crb2rgb(crb, r1, r2)
    #print('rgb', rgb.shape, rgb.max(), rgb.min())

    rgb_clamp = do_rgb_clamp(rgb) 
    print('rgb_clamp', rgb_clamp.max(), rgb_clamp.min())

    white_im = white_balance(rgb_clamp)

    plt.imshow(rgb)
    plt.figure()
    plt.imshow(rgb_clamp)
    plt.figure()
    plt.imshow(white_im)
    plt.show()
