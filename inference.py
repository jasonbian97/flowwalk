from argparse import ArgumentParser
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from einops import rearrange
import imageio
from torchvision import transforms
import sep_transforms
import glob
from flow_utils import flow_to_image, resize_flow
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
from Regressor import PWCLite
from collections import OrderedDict

def dict2cuda(inputs: dict):
    for k,v in inputs.items():
        if isinstance(v,torch.FloatTensor):
            inputs[k] = v.cuda()
        if isinstance(v, dict):
            inputs[k] = dict2cuda(v)
    return inputs

def inference(args, model):

    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()

    imgs_p = [glob.glob(args.input_dir + f"/**/*.{suffix}", recursive=True) for suffix in ["jpg","png","tiff"]]
    imgs_p = sorted(sum(imgs_p,[]))
    np8_inputs = [imageio.imread(p).astype(np.uint8) for p in imgs_p]
    np_shapes = [img.shape[:2] for img in np8_inputs]
    np32_inputs = [imageio.imread(p).astype(np.float32) for p in imgs_p]
    input_transform = transforms.Compose([
        sep_transforms.Zoom(*args.test_shape),
        sep_transforms.ArrayToTensor(),
        transforms.Normalize(mean=[0, 0, 0], std=[255, 255, 255]),
    ])
    inputs = [input_transform(i) for i in np32_inputs]
    rflows = []
    with torch.no_grad():
        for i in range(len(inputs)-1):
            print(f"pair: {i}")
            indict = {"img1": inputs[i].unsqueeze(0), "img2": inputs[i+1].unsqueeze(0)}
            indict = dict2cuda(indict) if torch.cuda.is_available() else indict

            img1, img2 = indict['img1'], indict['img2']
            img_pair_1st = torch.cat([img1, img2], 1)

            flows = model(img_pair_1st)['flows_fw']
            pred_flow = flows[0]
            rflows.append(pred_flow)

    resized_flows = [resize_flow(flow,shape) for flow,shape in zip(rflows,np_shapes[:-1])]

    vis_flows = [flow_to_image(rearrange(flow[0].detach().cpu().numpy(), "C H W -> H W C")) for flow in resized_flows]
    frames = []
    fig = plt.figure(tight_layout=True, figsize=(20, 5))
    gs = gridspec.GridSpec(1, 2)
    for j in range(len(vis_flows)):
        ax1 = fig.add_subplot(gs[0, 0])
        im1 = ax1.imshow(np8_inputs[j], animated=True)
        plt.axis('off')
        ax2 = fig.add_subplot(gs[0, 1])
        im2 = ax2.imshow(vis_flows[j], animated=True)
        plt.axis('off')
        frames.append([im1,im2])

    ani = animation.ArtistAnimation(fig, frames, interval=100, blit=True, repeat=False)
    ani.save(f'{args.input_dir}/flow_movie.mp4')

def _init_parser() -> ArgumentParser:
    parser = ArgumentParser()

    parser.add_argument( '--input_dir', type=str)
    parser.add_argument('--inference', type=int, default=1)
    parser.add_argument(
        '--pretrained_ckpt', type=str, default="",
        help=(''))
    parser.add_argument(
        '--test_shape',
        nargs="*",
        required=False,
        default=[448, 1024],
        type=int,
        help=""
    )
    return parser

from easydict import EasyDict

if __name__ == '__main__':
    parser = _init_parser()

    cfg = EasyDict({"n_frames": 2,
           "reduce_dense": True,
           "type": "pwclite",
           "upsample": True})

    args = parser.parse_args()

    model = PWCLite(cfg)
    pretrained_ckpt = args.pretrained_ckpt

    if args.pretrained_ckpt not in ["", None]:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if Path(pretrained_ckpt).exists():
            ckpt_path = pretrained_ckpt
            ckpt = torch.load(ckpt_path, map_location=torch.device(device))
            weights = ckpt['state_dict']
            # handle the scenario when keys do not exactly match
            new_weights = OrderedDict()
            model_keys = list(model.state_dict().keys())
            weight_keys = list(weights.keys())
            for a, b in zip(model_keys, weight_keys):
                new_weights[a] = weights[b]
            weights = new_weights
            model.load_state_dict(weights, strict=False)
        else:
            raise ValueError(f'Cannot find checkpoint {pretrained_ckpt} for model')

        inference(args, model)