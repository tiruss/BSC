from classifier import Classifier
import torch


def main():
    dummy = torch.rand((1, 3, 480, 480))
    classifier = Classifier()
    # classifier.load_state_dict(torch.load('weights/classifier.pth'))
    classifier.eval()

    output = classifier(dummy).softmax(dim=1)
    print(output)


if __name__ == "__main__":
    main()
