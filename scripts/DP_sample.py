"""
Like image_sample.py, but use a noisy image classifier to guide the sampling
process towards more realistic images.
"""
from PIL import Image as im
import argparse
import os
import datetime
import numpy as np
import torch as th
import matplotlib.pyplot as plt
import torch.distributed as dist
import torch.nn.functional as F

from guided_diffusion import sg_util, logger
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    classifier_defaults,
    create_model_and_diffusion,
    create_classifier,
    add_dict_to_argparser,
    args_to_dict,
)

def load_images(path : str) -> th.FloatTensor:
    ref_images = []
        
    for filename in os.listdir(path):
        im_path = os.path.join(path, filename)
        image = np.array(im.open(im_path))
        image = th.from_numpy(image).permute(2, 0, 1).float() / 255.0
        
        ref_images.append(image)
    
    ref_images = th.stack(ref_images)
    ref_images = (2 * ref_images - 1).to(sg_util.dev())
    return ref_images


def round_to_nearest_i_times_10x(scale):
    if scale == 0:
        return 0, 0
    
    exponent = int(np.floor(np.log10(scale)))
    coefficient = scale / (10 ** exponent)
    # print("coefficient:", coefficient)
    rounded_coefficient = round(coefficient)
    # print("rounded_coefficient:", rounded_coefficient)
    return rounded_coefficient, exponent


def get_finename(scale, iteration, prefix=""):
    if scale == 0:
        return f"{prefix}_iter={iteration}_s=0.pdf"
    # map tensor to float
    scale = scale.item()
    rounded_coefficient, exponent = round_to_nearest_i_times_10x(scale)
    return f"{prefix}_iter={iteration}_s={rounded_coefficient}e{exponent}.pdf" 

def save_images(results, num_rows, num_cols, filename, plot_dir):
    """
    Saves a batch of images and their corresponding samples to the specified directory.
    
    Args:
    results (dict): Dictionary containing the original images and their corresponding samples.
    num_rows (int): Number of rows in the plot.
    num_cols (int): Number of columns in the plot.
    filename (str): Filename for the saved plot.
    plot_dir (str): Directory to save the plots.    

    """

    os.makedirs(plot_dir, exist_ok=True)
    
    # Plot and save images
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(15, 15))
    keys = list(results.keys()) 
    for i in range(num_rows):
        data = results[keys[i]]
        data = ((data + 1) * 127.5).clamp(0, 255).to(th.uint8)
        data = data.permute(0, 2, 3, 1).contiguous().cpu().numpy()
        for j in range(num_cols):
            axs[i, j].imshow(data[j])
            axs[i, j].axis('off')
    
    
     
    for row in range(num_rows):
        if keys[row] == "x":
            axs[row, 0].text(-40, 128, 'data', rotation=90, fontsize=16, va='center')
        else:
            scale = keys[row].item()
            c, e = round_to_nearest_i_times_10x(scale) 
            axs[row, 0].text(-20, 128, f's={c}e{e}', rotation=90, fontsize=16, va='center') 
    
    plt.savefig(os.path.join(plot_dir, filename))
    plt.close(fig)


def main():
    args = create_argparser().parse_args()

    if args.log_dir: 
        log_dir_root = args.log_dir
    else: 
        log_dir_root = "logs";
     
    log_dir = os.path.join(
            log_dir_root,
            datetime.datetime.now().strftime("gdg-%Y-%m-%d-%H-%M-%S-%f"),
        ) 
    os.makedirs(log_dir, exist_ok=True) 
    logger.configure(dir=log_dir)


    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        sg_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(sg_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    ref_images = load_images(args.image_path)

    def cond_fn(x, t, y=None, s=None):
        return (ref_images - x) * s

    def model_fn(x, t, y=None, s=None):
        return model(x, t, s=s)

    plot_dir = os.path.join(log_dir, "plots")

    dp_scales = args.dp_scales
    dp_scales = th.tensor([float(x) for x in dp_scales.split(",")]) if dp_scales else th.tensor([0.0])
    diffusion.guide_schedule = th.ones((1000,)).to(sg_util.dev())
    
    results = {}    
    for i in range(args.num_iters):
        model_kwargs = {}

        for scale in dp_scales:
            logger.log(f"Sampling at scale {scale}")
            model_kwargs["s"] = scale.to(sg_util.dev())
            
            sample_fn = (
                diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
            )
            sample, diffusion_step = sample_fn(
                model_fn,
                (ref_images.size(0), 3, args.image_size, args.image_size),
                clip_denoised=args.clip_denoised,
                model_kwargs=model_kwargs,
                cond_fn=cond_fn,
                device=sg_util.dev(),
            )
            results[scale] = sample

        save_images(results=results,
                num_rows=len(dp_scales),
                num_cols=ref_images.size(0),
                filename=get_finename(0, i, "data"),
                plot_dir= plot_dir)
    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_iters=1,
        log_dir="",
        use_ddim=False,
        model_path="",
        image_path="",
        dp_scales="",
    )
    defaults.update(model_and_diffusion_defaults())
    defaults.update(classifier_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
