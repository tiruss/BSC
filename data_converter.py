import cv2
import os
import numpy as np
import glob
import argparse

class BlurGenerator:
    def __init__(self, input_img):
        self.input_img = input_img

    def gaussian_blur(self, kernel_size=5, sigma=1.5):
        img = cv2.imread(self.input_img)
        blur = cv2.GaussianBlur(img, (kernel_size, kernel_size), sigma)
        # cv2.imwrite(os.path.join(self.output_dir, 'gaussian_blur.jpg'), blur)
        return blur

    def motion_blur(self, kernel_size=15):
        img = cv2.imread(self.input_img)
        ratio = np.random.uniform()
        # Generate the motion kernel
        kernel_motion_blur = np.zeros((kernel_size, kernel_size))
        if ratio < 0.5:  # Horizontal
            kernel_motion_blur[int((kernel_size-1)/2), :] = np.ones(kernel_size)
        else:  # Vertical
            kernel_motion_blur[:, int((kernel_size - 1) / 2)] = np.ones(kernel_size)
        kernel_motion_blur = kernel_motion_blur / kernel_size
        # Apply the kernel
        blur = cv2.filter2D(img, -1, kernel_motion_blur)

        return blur

    def make_blur(self):
        ratio = np.random.uniform()
        if ratio < 0.5:
            return self.gaussian_blur()
        else:
            return self.motion_blur()

    def save_img(self, img, output_dir, img_name):
        cv2.imwrite(os.path.join(output_dir, img_name), img)


def extract_frames(video_path, output_dir, sampling_rate=30):
    vidcap = cv2.VideoCapture(video_path)
    vid_name = os.path.basename(video_path).split('.')[0]
    success, image = vidcap.read()
    count = 0
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "clear"), exist_ok=True)
    # os.makedirs(os.path.join(output_dir, "clear", vid_name), exist_ok=True)

    while success:
        count += 1
        if count % sampling_rate == 0:
            cv2.imwrite(f"{output_dir}/clear/{vid_name}_{count}.jpg", image)
        success, image = vidcap.read()
        print('Read a new frame: ', success)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='demo_data/0725_IncompleteFocus_Bad_001.mp4')
    parser.add_argument('--output', type=str, default='results')
    parser.add_argument('--mode', type=str, default='frames', help="frames, blur")

    args = parser.parse_args()
    input_video = args.input
    output_dir = args.output
    mode = args.mode
    # blur_generator = BlurGenerator("car.jpg")
    # blur = blur_generator.motion_blur()
    # cv2.imwrite("gaussian_blur.jpg", blur)
    if mode == "frames":
        extract_frames(input_video, output_dir)
    elif mode == "blur":
        img_dir = glob.glob(os.path.join(output_dir, "clear", "*.jpg"))
        output_dir = os.path.join(output_dir, "blur")
        for img in img_dir:
            blur_generator = BlurGenerator(img)
            blur = blur_generator.make_blur()
            cv2.imwrite(os.path.join(output_dir, os.path.basename(img)), blur)
