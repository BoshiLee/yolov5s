from torch import onnx
import torch


def run(weights=r'v5s-p6_wider_face.pt',  # model.pt path(s)
        imgsz=640,  # inference size (pixels)
        device=1,  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        ):

    # Load model
    model = torch.load(weights)['model']
    # Input to the model
    x = torch.randn(1, 3, imgsz, imgsz, requires_grad=True)
    model.train(False)
    # Export the model
    torch_out = onnx._export(model,  # model being run
                                   x,  # model input (or a tuple for multiple
                                   f='v5s-p6_wider_face.onnx',
                                   export_params=True)




if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description='PyTorch sphereface lfw')
    # parser.add_argument('--net', '-n', default='sphere20a', type=str)
    # parser.add_argument('--model', '-m', default=r'model/sphere20a_20171020.pth', type=str)

    run()

