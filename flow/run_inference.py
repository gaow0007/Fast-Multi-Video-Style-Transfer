import argparse
from path import Path
from torchvision.utils import save_image
import torch
import torch.backends.cudnn as cudnn
import models
from tqdm import tqdm
import torchvision.transforms as transforms
import flow_transforms
from scipy.ndimage import imread
from scipy.misc import imsave
import numpy as np
from main import flow2rgb
from PIL import Image


def test_transform():
    transform_list = [
           # transforms.Resize(size=(512, 512)),
            transforms.ToTensor()]
    transform = transforms.Compose(transform_list)
    return transform


model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__"))


parser = argparse.ArgumentParser(description='PyTorch FlowNet inference on a folder of img pairs',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--data', metavar='DIR',
                    help='path to images folder, image names must match \'[name]0.[ext]\' and \'[name]1.[ext]\'')
parser.add_argument('--pretrained', metavar='PTH', help='path to pre-trained model')
parser.add_argument('--output', metavar='DIR', default=None,
                    help='path to output folder. If not set, will be created in data folder')
parser.add_argument('--div-flow', default=20, type=float,
                    help='value by which flow will be divided. overwritten if stored in pretrained file')
parser.add_argument("--img-exts", default=['png', 'jpg', 'bmp'], nargs='*', type=str, help="images extensions to glob")
parser.add_argument('--max_flow', default=None, type=float,
                    help='max flow value. Flow map color is saturated above this value. If not set, will use flow map\'s max value')
parser.add_argument('--upsampling', '-u', choices=['nearest', 'bilinear'], default=None, help='if not set, will output FlowNet raw input,'
                    'which is 4 times downsampled. If set, will output full resolution flow map, with selected upsampling')
parser.add_argument('--bidirectional', action='store_true', help='if set, will output invert flow (from 1 to 0) along with regular flow')

@torch.no_grad()
def main():
    global args, save_path
    args = parser.parse_args()
    data_dir = Path(args.data)
    print("=> fetching img pairs in '{}'".format(args.data))
    save_path = data_dir/'flow'
    print('=> will save everything to {}'.format(save_path))
    save_path.makedirs_p()

    # Data loading code
    input_transform = transforms.Compose([
        flow_transforms.ArrayToTensor(),
        transforms.Normalize(mean=[0,0,0], std=[255,255,255]),
        transforms.Normalize(mean=[0.411,0.432,0.45], std=[1,1,1])
    ])

    img_pairs = []
    for ext in args.img_exts:
        test_files = data_dir.files('*0.{}'.format(ext))
        for file in test_files:
            img_pair = file.parent / (file.namebase[:-1] + '1.{}'.format(ext))
            if img_pair.isfile():
                img_pairs.append([file, img_pair])

    print('{} samples found'.format(len(img_pairs)))
    # create model
    network_data = torch.load(args.pretrained)
    print("=> using pre-trained model '{}'".format(network_data['arch']))
    model = models.__dict__[network_data['arch']](network_data).cuda()
    model.eval()
    cudnn.benchmark = True

    if 'div_flow' in network_data.keys():
        args.div_flow = network_data['div_flow']
    input_transform = test_transform()

    for (img1_file, img2_file) in tqdm(img_pairs):

        # img1 = input_transform(imread(img1_file))
        # img2 = input_transform(imread(img2_file))
        img1 = Image.open(img1_file).convert('RGB')
        img1 = input_transform(img1)

        img2 = Image.open(img2_file).convert('RGB')
        img2 = input_transform(img2)


        input_var = torch.tensor(torch.cat([img1, img2]).cuda()).unsqueeze(0)

        if args.bidirectional:
            # feed inverted pair along with normal pair
            inverted_input_var = torch.tensor(torch.cat([img2, img1],0).cuda()).unsqueeze(0)
            input_var = torch.cat([input_var, inverted_input_var])

        # compute output
        output = model(input_var)
        args.upsampling = 'bilinear'
        if args.upsampling is not None:
            output = torch.nn.functional.upsample(output, size=img1.size()[-2:], mode=args.upsampling)  * 80
            # output = output.permute(0, 2, 3, 1) * 4
            # img1 = torch.nn.functional.avg_pool2d(img1, 4)
            # img2 = torch.nn.functional.avg_pool2d(img2, 4)
        # N, C, H, W = output.size()
        print(torch.sum(output > 3), flush=True)
        from PWCNet import warp
        '''
        output = torch.zeros(N, C, H, W)
        output.fill_(10.0)
        output = output.cuda()
        '''

        result = warp(img2.unsqueeze(0).cuda(), 1.0 * output)
        print(result.size())
        save_image(result, 'img2to.jpg')
        save_image(img1, 'img1.jpg')
        save_image(img2, 'img2.jpg')

        for suffix, flow_output in zip(['flow', 'inv_flow'], output.data.cpu()):
            rgb_flow = flow2rgb(args.div_flow * flow_output, max_value=args.max_flow)
            to_save = (rgb_flow * 255).astype(np.uint8).transpose(1,2,0)
            imsave(save_path/'{}{}.png'.format(img1_file.namebase[:-1], suffix), to_save)


if __name__ == '__main__':
    main()
