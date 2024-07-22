
import os
import argparse
import datetime
import torch as th
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.utils.data import DataLoader
from guided_diffusion import sg_util, logger
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
    def cond_fn(x, t, y=None, s=None):
        return (images - x) * s
    
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

    clf = models.resnet50(weights = models.ResNet50_Weights.DEFAULT).to(sg_util.dev())
    clf.eval()

    base_correct = 0
    total = 0

    eval_set = [i for i in val_loader][:args.batch_number]

    with th.no_grad():
        logger.log("Measuring base performance...")
        for images, labels in eval_set:
            images = images.to(sg_util.dev())
            outputs = clf(images).to('cpu')
            _, predicted = th.max(outputs.data, 1)
            total += labels.size(0)
            base_correct += (predicted == labels).sum().item()

        accuracy = 100 * base_correct / total
        logger.log(f'Accuracy of the network on the ImageNet validation images: {accuracy:.2f}%')
        
        scales = [float(i) for i in args.guide_scales.split(",")]
        for scale in scales:
            logger.log(f"Measuring performance at scale {scale}...")
            model_kwargs = {"s" : scale}
            for images, labels in eval_set:
                correct = 0
                images = upscale(images)
                images = images.to(sg_util.dev())                
                samples, _ = diffusion.p_sample_loop(
                    model_fn,
                    (images.size(0), 3, 256, 256),
                    clip_denoised=args.clip_denoised,
                    model_kwargs=model_kwargs,
                    cond_fn=cond_fn,
                    device=sg_util.dev(),                
                )
                
                images = downscale(samples)
                outputs = clf(images).to('cpu')
                _, predicted = th.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()

            accuracy = 100 * correct / total
            logger.log(f'Accuracy of the network after {scale} strength guiding: {accuracy:.2f}%')

def create_argparser():
    defaults = dict(
        clip_denoised = True,
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
