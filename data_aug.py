import cv2
import numpy as np

def sp_noise_img(image, prob):
    '''
    添加椒盐噪声
    prob:噪声比例
    '''
    output = np.zeros(image.shape, np.uint8)
    thres = 1 - prob
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = np.random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output


def contrast_bright_img(img, alpha, beta):
    blank = np.zeros(img.shape, img.dtype)
    # dst = alpha * img + beta * blank
    alpha_ = np.random.uniform(alpha, 1)
    beta_ = np.random.uniform(0, beta)
    img_cb = cv2.addWeighted(img, alpha_, blank, 1-alpha_, beta_)
    return img_cb

def rotate_img(img, angle):
    H, W = img.shape[:2]
    angle_ = np.random.randint(-angle,angle)
    cX, cY = W // 2, H // 2
    M = cv2.getRotationMatrix2D((cX, cY), -angle_, 1.0)
    rotate_img = cv2.warpAffine(img, M, (W, H))
    return rotate_img

#
# image = cv2.imread('./data/my_data/train/000045.png')
# cv2.imshow('1', image)
#
# # ga_img = sp_noise_img(image, 0.05)
# # ga_img = contrast_bright_img(image, 0.25, 0.5)
# ga_img = rotate_img(image, 30)
# cv2.imshow('2', ga_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()