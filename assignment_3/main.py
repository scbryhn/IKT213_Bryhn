import cv2
import os

OUTPUT_DIR = "solutions"

def sobel_edge_detection(image):
    imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    imageBlur = cv2.GaussianBlur(imageGray, ksize=(3,3), sigmaX=0)
    sobelxy = cv2.Sobel(src=imageBlur, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=1)
    cv2.imwrite(f"{OUTPUT_DIR}/Part_1.png", sobelxy)




def main():
    os.makedirs(f"{OUTPUT_DIR}", exist_ok=True)
    image = cv2.imread("lambo.png", cv2.IMREAD_UNCHANGED)
    sobel_edge_detection(image)

if __name__ == "__main__":
    main()