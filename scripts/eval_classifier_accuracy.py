import os
import gc
import argparse
import datetime
import torch as th
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.utils.data import DataLoader
from guided_diffusion import sg_util, logger
from copy import deepcopy
from guided_diffusion.script_util import (
    NUM_CLASSES,
    create_model_and_diffusion,
    add_dict_to_argparser,
)

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

    guide_schedule = th.ones((1000,)).to(sg_util.dev())

    model, diffusion = create_model_and_diffusion(
        image_size=256,
        class_cond=False,
        learn_sigma=True,
        num_channels=256,
        num_res_blocks=2,
        num_head_channels=64,
        attention_resolutions="32,16,8",
        diffusion_steps=1000,
        noise_schedule="linear",
        timestep_respacing=[250],
        use_scale_shift_norm=True,
        resblock_updown=True,
        use_fp16=args.use_fp16,
        use_new_attention_order=True,
        channel_mult="",
        num_heads=4,
        num_heads_upsample=-1,
        dropout=0.0,
        use_kl=False,
        predict_xstart=False,
        rescale_timesteps=False,
        rescale_learned_sigmas=False,
        use_checkpoint=False

    )
    diffusion.guide_schedule = guide_schedule
    model.load_state_dict(
        sg_util.load_state_dict("models/256x256_diffusion_uncond.pt", map_location="cpu")
    )
    model.to(sg_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    def model_fn(x, t, y=None, s=None):
        return model(x, t, s=s)
    
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    upscale = transforms.Resize(256)
    downscale = transforms.Resize(224)

    val_dataset = datasets.ImageNet(root='./imagenet', split='val', transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    clf = models.vit_b_16(weights = models.ViT_B_16_Weights.DEFAULT).to(sg_util.dev())
    clf.eval()

    base_correct = 0
    total = 0

    eval_set = [i for i in val_loader][:args.batch_number]

    with th.no_grad():
        logger.log("Measuring base performance...")
        for images, labels in eval_set:
            img = images.to(sg_util.dev())
            outputs = clf(img).to('cpu')
            _, predicted = th.max(outputs.data, 1)
            total += labels.size(0)
            base_correct += (predicted == labels).sum().item()

            del _, img, outputs, predicted
            th.cuda.empty_cache()
            gc.collect

        accuracy = 100 * base_correct / total
        logger.log(f'Accuracy of the network on the ImageNet validation images: {accuracy:.2f}%')
        
        scales = [float(i) for i in args.guide_scales.split(",")]
        for scale in scales:
            logger.log(f"Measuring performance at scale {scale}...")
            model_kwargs = {"s" : scale}
            for images, labels in eval_set:
                correct = 0
                img = upscale(images)
                img = img.to(sg_util.dev())
                
                diff_img = scale_imagenet_to_diffusion(img)

                def cond_fn(x, t, y=None, s=1.0):
                    return (diff_img - x) * s     
                           
                samples, _ = diffusion.p_sample_loop(
                    model_fn,
                    (img.size(0), 3, 256, 256),
                    clip_denoised=args.clip_denoised,
                    model_kwargs=model_kwargs,
                    cond_fn=cond_fn,
                    device=sg_util.dev(),                
                )
                
                img = scale_diffusion_to_imagenet(downscale(samples))
                outputs = clf(img).to('cpu')
                _, predicted = th.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()

                del samples, _, img, outputs, predicted
                th.cuda.empty_cache()
                gc.collect()

            accuracy = 100 * correct / total
            logger.log(f'Accuracy of the network after {scale} strength guiding: {accuracy:.2f}%')

def scale_imagenet_to_diffusion(
        img : th.FloatTensor, 
        mean = [0.485, 0.456, 0.406],
        std = [0.229, 0.224, 0.225]):
    
    result = deepcopy(img)
    result[:, 0] = result[:, 0] * std[0] + mean[0]
    result[:, 1] = result[:, 1] * std[1] + mean[1]
    result[:, 2] = result[:, 2] * std[2] + mean[2]
    return 2 * result - 1

def scale_diffusion_to_imagenet(
        img : th.FloatTensor, 
        mean = [0.485, 0.456, 0.406],
        std = [0.229, 0.224, 0.225]):
    
    result = .5 * img + .5
    result[:, 0] = (result[:, 0] - mean[0])/std[0]
    result[:, 1] = (result[:, 1] - mean[1])/std[1]
    result[:, 2] = (result[:, 2] - mean[2])/std[2]
    return result

def create_argparser():
    defaults = dict(
        clip_denoised = False,
        guide_scales = "0.4,0.8",
        guide_profile = "constant",
        use_fp16 = False,
        log_dir = "logs",
        batch_size = 32,
        batch_number = 10
    )

    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

if __name__ == '__main__':
    main()
