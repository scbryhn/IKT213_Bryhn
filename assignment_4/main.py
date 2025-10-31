import cv2
import numpy as np

IMAGE_FOLDER = r"images"
RESULT_FOLDER = r"results"

def find_corner(reference_image):
    gray = cv2.cvtColor(reference_image,cv2.COLOR_BGR2GRAY)

    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray, 2, 3, 0.04)
    dst = cv2.dilate(dst, None)

    img = reference_image.copy()
    threshold = 0.1 * dst.max()
    corners = np.where(dst > threshold)
    img[corners] = [0, 0, 255]

    cv2.imwrite(f"{RESULT_FOLDER}/harris.png", img)

def align_images_sift(image_to_align, reference_image, max_features=10, good_match_percent=0.7):
    img1 = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(image_to_align, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create(nfeatures=max_features)

    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1, des2, k=2)

    good_matches = []
    for m1, m2 in matches:
        if m1.distance < good_match_percent * m2.distance:
            good_matches.append(m1)

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    match_img = cv2.drawMatches(reference_image, kp1, image_to_align, kp2, good_matches, None, flags=2)
    cv2.imwrite(f"{RESULT_FOLDER}/matches.jpg", match_img)


    if len(good_matches) < 4:
        print("Not enough matches for homography")
        return

    H, status = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)

    if H is None:
        print("Homography could not be computed.")
        return

    h, w, _ = reference_image.shape
    aligned = cv2.warpPerspective(image_to_align, H, (w, h))

    cv2.imwrite(f"{RESULT_FOLDER}/aligned.jpg", aligned)

if __name__ == "__main__":
    reference_image = cv2.imread(f"{IMAGE_FOLDER}/reference_img.png")
    align_this = cv2.imread(f"{IMAGE_FOLDER}/align_this.jpg")

    find_corner(reference_image)
    align_images_sift(align_this, reference_image, max_features=5000)