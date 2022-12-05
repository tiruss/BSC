import cv2
import os
import numpy as np

class BlurGenerator:
    def __init__(self, input_img):
        self.input_img = input_img

    def gaussian_blur(self, kernel_size=5, sigma=1.5):
        img = cv2.imread(self.input_img)
        blur = cv2.GaussianBlur(img, (kernel_size, kernel_size), sigma)
        # cv2.imwrite(os.path.join(self.output_dir, 'gaussian_blur.jpg'), blur)
        return blur

    def motion_blur(self, kernel_size=15, angle=45):
        img = cv2.imread(self.input_img)
        # Generate the motion kernel
        kernel_motion_blur = np.zeros((kernel_size, kernel_size))
        kernel_motion_blur[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)
        kernel_motion_blur = kernel_motion_blur / kernel_size
        # Apply the kernel
        blur = cv2.filter2D(img, -1, kernel_motion_blur)

        return blur


def extract_frames(video_path, output_dir, frame_rate=30):
    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    count = 0
    os.makedirs(output_dir, exist_ok=True)

    while success:
        count += 1
        if count % frame_rate == 0:
            cv2.imwrite(f"{output_dir}/frame{count}.jpg", image)
        success, image = vidcap.read()
        print('Read a new frame: ', success)




if __name__ == "__main__":
    blur_generator = BlurGenerator("car.jpg")
    blur = blur_generator.motion_blur()
    cv2.imwrite("gaussian_blur.jpg", blur)
    # extract_frames("demo_data/0725_IncompleteFocus_Bad_001.mp4", "frames")
