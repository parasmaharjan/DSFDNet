import numpy as np
import pywt
from scipy.ndimage import filters

def decompose3ch(img, filter='haar'):
    LL0, (LH0, HL0, HH0) = pywt.dwt2(img[0, :, :, 0], filter)
    LL1, (LH1, HL1, HH1) = pywt.dwt2(img[0, :, :, 1], filter)
    LL2, (LH2, HL2, HH2) = pywt.dwt2(img[0, :, :, 2], filter)

    LLL0, (LLH0, LHL0, LHH0) = pywt.dwt2(LL0, filter)
    LLL1, (LLH1, LHL1, LHH1) = pywt.dwt2(LL1, filter)
    LLL2, (LLH2, LHL2, LHH2) = pywt.dwt2(LL2, filter)

    LLL = np.expand_dims(np.stack((LLL0, LLL1, LLL2), axis=2), axis=0)
    LLH = np.expand_dims(np.stack((LLH0, LLH1, LLH2), axis=2), axis=0)
    LHL = np.expand_dims(np.stack((LHL0, LHL1, LHL2), axis=2), axis=0)
    LL = np.expand_dims(np.stack((LL0, LL1, LL2), axis=2), axis=0)
    LH = np.expand_dims(np.stack((LH0, LH1, LH2), axis=2), axis=0)
    HL = np.expand_dims(np.stack((HL0, HL1, HL2), axis=2), axis=0)
    HH = np.expand_dims(np.stack((HH0, HH1, HH2), axis=2), axis=0)

    return LLL, LLH, LHL, LL, LH, HL, HH

def decompose4ch(img, filter='haar', gaus=False):
    LL0, (LH0, HL0, HH0) = pywt.dwt2(img[0, :, :, 0], filter)
    LL1, (LH1, HL1, HH1) = pywt.dwt2(img[0, :, :, 1], filter)
    LL2, (LH2, HL2, HH2) = pywt.dwt2(img[0, :, :, 2], filter)
    LL3, (LH3, HL3, HH3) = pywt.dwt2(img[0, :, :, 3], filter)

    LLL0, (LLH0, LHL0, LHH0) = pywt.dwt2(LL0, filter)
    LLL1, (LLH1, LHL1, LHH1) = pywt.dwt2(LL1, filter)
    LLL2, (LLH2, LHL2, LHH2) = pywt.dwt2(LL2, filter)
    LLL3, (LLH3, LHL3, LHH3) = pywt.dwt2(LL3, filter)

    LLL = np.expand_dims(np.stack((LLL0, LLL1, LLL2, LLL3), axis=2), axis=0)
    LLH = np.expand_dims(np.stack((LLH0, LLH1, LLH2, LLH3), axis=2), axis=0)
    LHL = np.expand_dims(np.stack((LLH0, LHL1, LHL2, LHL3), axis=2), axis=0)
    LL = np.expand_dims(np.stack((LL0, LL1, LL2, LL3), axis=2), axis=0)
    LH = np.expand_dims(np.stack((LH0, LH1, LH2, LH3), axis=2), axis=0)
    HL = np.expand_dims(np.stack((HL0, HL1, HL2, HL3), axis=2), axis=0)
    HH = np.expand_dims(np.stack((HH0, HH1, HH2, HH3), axis=2), axis=0)

    if gaus:
        LL_  = filters.gaussian_filter(LL,sigma=0.5)
        LH_ = filters.gaussian_filter1d(LH,0.5,1)
        HL_ = filters.gaussian_filter1d(HL,0.5,0)

    return LLL, LLH, LHL, LL, LH, HL, HH

def combine4ch(LLL, LLH, LHL, LL, LH, HL, HH, filter='haar'):
    LL0 = LL[0, :, :, 0]
    LL1 = LL[0, :, :, 1]
    LL2 = LL[0, :, :, 2]
    LL3 = LL[0, :, :, 3]

    LH0 = LH[0, :, :, 0]
    LH1 = LH[0, :, :, 1]
    LH2 = LH[0, :, :, 2]
    LH3 = LH[0, :, :, 3]

    HL0 = HL[0, :, :, 0]
    HL1 = HL[0, :, :, 1]
    HL2 = HL[0, :, :, 2]
    HL3 = HL[0, :, :, 3]

    HH0 = HH[0, :, :, 0]
    HH1 = HH[0, :, :, 1]
    HH2 = HH[0, :, :, 2]
    HH3 = HH[0, :, :, 3]

    # LLL0 = LLL[0, :, :, 0]
    # LLL1 = LLL[0, :, :, 1]
    # LLL2 = LLL[0, :, :, 2]
    # LLL3 = LLL[0, :, :, 3]
    #
    # LLH0 = LLH[0, :, :, 0]
    # LLH1 = LLH[0, :, :, 1]
    # LLH2 = LLH[0, :, :, 2]
    # LLH3 = LLH[0, :, :, 3]
    #
    # temp0 = np.zeros_like(LLL0)
    # temp1 = np.zeros_like(LLL0)
    # temp2 = np.zeros_like(LLL0)
    # temp3 = np.zeros_like(LLL0)
    #
    # coef00 = LLL0, (LLH0, temp0, temp0)
    # coef01 = LLL1, (LLH1, temp1, temp1)
    # coef02 = LLL2, (LLH2, temp2, temp2)
    # coef03 = LLL3, (LLH3, temp3, temp3)
    #
    # LL0 = pywt.idwt2(coef00, filter)
    # LL1 = pywt.idwt2(coef01, filter)
    # LL2 = pywt.idwt2(coef02, filter)
    # LL3 = pywt.idwt2(coef03, filter)

    coef0 = LL0, (LH0, HL0, HH0)
    coef1 = LL1, (LH1, HL1, HH1)
    coef2 = LL2, (LH2, HL2, HH2)
    coef3 = LL3, (LH3, HL3, HH3)

    img0 = pywt.idwt2(coef0, filter)
    img1 = pywt.idwt2(coef1, filter)
    img2 = pywt.idwt2(coef2, filter)
    img3 = pywt.idwt2(coef3, filter)
    return np.stack((img0, img1, img2, img3), axis=2)

def combine1ch(LLL, LLH, LHL, LL, LH, HL, HH, filter='haar'):
    LL0 = LL[0, :, :, 0]

    LH0 = LH[0, :, :, 0]

    HL0 = HL[0, :, :, 0]

    HH0 = HH[0, :, :, 0]

    coef0 = LL0, (LH0, HL0, HH0)

    img0 = pywt.idwt2(coef0, filter)
    return img0