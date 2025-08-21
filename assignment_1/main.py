import cv2

def print_image_information(image):
    height, width, channels = image.shape

    print("Height:\t\t", height)
    print("Width:\t\t", width)
    print("Channels:\t", channels)
    print("Size:\t\t", image.size)
    print("Data type:\t", image.dtype)

def print_camera_information():
    print("Hello")
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        return

    fps = cam.get(cv2.CAP_PROP_FPS)
    frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

    with open("solutions/camera_outputs.txt", "w") as file:
        file.write(f"fps:\t{fps}\n")
        file.write(f"height:\t{frame_height}\n")
        file.write(f"width:\t{frame_width}\n")

def main():
    image = cv2.imread("pictures/lena-1.png", cv2.IMREAD_UNCHANGED)
    print_image_information(image)
    print_camera_information()

if __name__ == '__main__':
    main()