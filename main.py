from classifier import Classifier
import torch
import cv2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main(input_video, output_dir, model_path):

    global output
    classifier = torch.nn.DataParallel(Classifier())
    classifier.load_state_dict(torch.load(model_path))
    classifier.eval().to(device)

    cap = cv2.VideoCapture(input_video)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (224, 224))
            frame = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).float().to(device)
            output = classifier(frame)

            print(output)
            # prediction = torch.max(output, 1)[1]
            # print(prediction)
        else:
            break

    return output

if __name__ == "__main__":
    prediction = main("/home/dk/BSC/221201/01_Full/221117_01.mp4", "data", "weights/classifier.pth")

