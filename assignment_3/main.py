import cv2
import os
import numpy as np

INPUT_DIR = "images"
OUTPUT_DIR = "solutions"

def sobel_edge_detection(image):
    imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    imageBlur = cv2.GaussianBlur(imageGray, ksize=(3,3), sigmaX=0)
    sobelxy = 255 * cv2.Sobel(src=imageBlur, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=1).clip(0, 255).astype(np.uint8)
    cv2.imwrite(f"{OUTPUT_DIR}/Part_1.png", sobelxy)

def canny_edge_detection(image, threshold_1, threshold_2):
    imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    imageBlur = cv2.GaussianBlur(imageGray, ksize=(3,3), sigmaX=0)
    edges = cv2.Canny(imageBlur, threshold_1, threshold_2)
    cv2.imwrite(f"{OUTPUT_DIR}/Part_2.png", edges)

def template_match(image, template):
    h,w, _ = template.shape
    imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    templateGray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    res = cv2.matchTemplate(imageGray,templateGray,cv2.TM_CCOEFF_NORMED)
    threshold = 0.9
    loc = np.where(res >= threshold)
    for x,y in zip(*loc[::-1]):
        cv2.rectangle(image, (x,y), (x + w, y + h), (0,0,255), 1)
    cv2.imwrite(f"{OUTPUT_DIR}/Part_3.jpg", image)

def resize( image, scale_factor: int, up_or_down: str):
    rows, cols, _ = image.shape

    if up_or_down == "up":
        resizedImage = cv2.pyrUp(image, dstsize=(scale_factor * cols, scale_factor * rows))
    elif up_or_down== "down":
        resizedImage = cv2.pyrDown(image, dstsize=(scale_factor // cols, scale_factor // rows))
    else:
        print("function resize: Invalid input")
        return
    cv2.imwrite(f"{OUTPUT_DIR}/Part_4.png", resizedImage)

def main():
    os.makedirs(f"{OUTPUT_DIR}", exist_ok=True)
    lamboImage = cv2.imread(f"{INPUT_DIR}/lambo.png", cv2.IMREAD_UNCHANGED)
    shapesImageGray = cv2.imread(f"{INPUT_DIR}/shapes-1.png", cv2.IMREAD_UNCHANGED)
    shapesTemplateGray = cv2.imread(f"{INPUT_DIR}/shapes_template.jpg", cv2.IMREAD_UNCHANGED)

    sobel_edge_detection(lamboImage)
    canny_edge_detection(lamboImage, threshold_1=50, threshold_2=50)

    template_match(shapesImageGray, shapesTemplateGray)
    resize(lamboImage, scale_factor=2, up_or_down="up")
    return 0

if __name__ == "__main__":
    main()