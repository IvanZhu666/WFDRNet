import argparse
import os
import shutil
import warnings

import torch
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torchmetrics import AUROC, AveragePrecision
from PIL import Image, ImageDraw, ImageFont
from torchvision.transforms import ToPILImage

import numpy as np
import matplotlib.pyplot as plt

from constant import RESIZE_SHAPE, NORMALIZE_MEAN, NORMALIZE_STD, ALL_CATEGORY
from data.mvtec_dataset import MVTecDataset
from model.WFDRNet_Model import WFDRNet

warnings.filterwarnings("ignore")

total_detail = {}
normalize_mean = [0.485, 0.456, 0.406]
normalize_std = [0.229, 0.224, 0.225]


def unnormalize(tensor, mean, std):
    single_channel_mean = torch.tensor(mean).view(-1, 1, 1)
    single_channel_std = torch.tensor(std).view(-1, 1, 1)

    if tensor.dim() == 4:
        batch_mean = single_channel_mean.view(1, -1, 1, 1)
        batch_std = single_channel_std.view(1, -1, 1, 1)
    else:
        batch_mean = single_channel_mean
        batch_std = single_channel_std

    unnormalized_tensor = tensor * batch_std + batch_mean
    return unnormalized_tensor


def visualize_and_save_images(
    args,
    results_mask_dir,
    imgs,
    mask_maps,
    heart_maps,
    predicted_masks,
    littlename,
    category,
    global_min,
    global_max
):
    tensor_to_pil = ToPILImage()
    font = ImageFont.load_default()

    for i in range(len(results_mask_dir)):
        pre_mask = predicted_masks[i].cpu().numpy().astype(np.uint8)
        heart_map = heart_maps[i].cpu().numpy().astype(np.uint8)
        mask_map = mask_maps[i].cpu().numpy().astype(np.uint8)

        image = imgs[i].cpu()
        image = unnormalize(image, normalize_mean, normalize_std)
        image = tensor_to_pil(image)

        pre_mask = Image.fromarray(pre_mask[0] * 255)
        mask_map = Image.fromarray(mask_map[0] * 255)

        out_file_path = f"{args.results_mask}/{category}/{results_mask_dir[i]}"
        filename = os.path.basename(out_file_path)

        out_directory_path = os.path.dirname(out_file_path)
        if not os.path.exists(out_directory_path):
            os.makedirs(out_directory_path)

        heart_map_array = heart_map.squeeze()
        heart_map_array = (heart_map_array - global_min) / (global_max - global_min + 1e-8)

        plt.figure(num=1, figsize=(256 / 100, 256 / 100))
        cnorm = plt.Normalize(vmin=0, vmax=1)
        plt.imshow(heart_map_array, norm=cnorm, cmap="jet")
        plt.axis("off")
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
        plt.margins(0, 0)

        temp_heatmap_file = f"{out_directory_path}/temp_heatmap_{filename}"
        plt.savefig(temp_heatmap_file, bbox_inches="tight", dpi=100, pad_inches=0)
        plt.close()

        heatmap_img = Image.open(temp_heatmap_file)

        combined_width = mask_map.width + pre_mask.width + heatmap_img.width + image.width
        combined_height = max(mask_map.height, pre_mask.height, heatmap_img.height)
        combined_image = Image.new("RGB", (combined_width, combined_height), color=(250, 250, 250))

        combined_image.paste(image, (0, 0))
        combined_image.paste(mask_map, (image.width, 0))
        combined_image.paste(heatmap_img, (image.width + mask_map.width, 0))
        combined_image.paste(pre_mask, (image.width + mask_map.width + heatmap_img.width, 0))

        draw = ImageDraw.Draw(combined_image)
        labels = ["Image", "ground truth", "Heat Map", "Predicted Mask"]
        label_positions = [
            (0, 0),
            (image.width, 0),
            (image.width + mask_map.width, 0),
            (image.width + mask_map.width + heatmap_img.width, 0)
        ]

        label_colors = [(255, 0, 0)] * len(labels)
        for label, pos, color in zip(labels, label_positions, label_colors):
            draw.text(pos, label, fill=color, font=font)

        combined_file_path = f"{out_directory_path}/{littlename}_{filename}"
        combined_image.save(combined_file_path)

        os.remove(temp_heatmap_file)

def evaluate_single_model(args, category, model, visualizer, global_step=0):

    model.eval()
    with torch.no_grad():
        dataset = MVTecDataset(
            is_train=False,
            mvtec_dir=os.path.join(args.mvtec_path, category, "test"),
            resize_shape=RESIZE_SHAPE,
            normalize_mean=NORMALIZE_MEAN,
            normalize_std=NORMALIZE_STD,
        )

        dataloader = DataLoader(dataset, batch_size=args.bs, shuffle=False, num_workers=args.num_workers)

        WFDRNet_AUROC = AUROC().cuda()
        WFDRNet_AP = AveragePrecision().cuda()
        WFDRNet_Image_AUROC = AUROC().cuda()

        First_AUROC = AUROC().cuda()
        First_AP = AveragePrecision().cuda()
        First_Image_AUROC = AUROC().cuda()

        imgs = []
        mask_maps = []
        heart_maps_WFDRNet = []
        predicted_masks = []

        for _, sample_batched in enumerate(dataloader):
            img = sample_batched["img"].cuda()
            mask = sample_batched["mask"].to(torch.int64).cuda()

            predicted_mask, output_first, first_mask = model(img)

            predicted_mask = F.interpolate(
                predicted_mask,
                size=mask.size()[2:],
                mode="bilinear",
                align_corners=False,
            )

            first_mask = F.interpolate(
                first_mask, size=mask.size()[2:], mode="bilinear", align_corners=False
            )

            if global_step == 0:
                for i in range(predicted_mask.shape[0]):
                    imgs.append(img[i])
                    mask_maps.append(mask[i])
                    heart_maps_WFDRNet.append(predicted_mask[i] * 50)
                    predicted_masks.append(torch.where(predicted_mask[i] >= args.mask_thresh, 1, 0))

            mask_sample = torch.max(mask.view(mask.size(0), -1), dim=1)[0]
            predicted_mask_sample, _ = torch.sort(predicted_mask.view(predicted_mask.size(0), -1), dim=1, descending=True)
            predicted_mask_sample = torch.mean(predicted_mask_sample[:, : args.T], dim=1)

            first_mask_sample, _ = torch.sort(first_mask.view(first_mask.size(0), -1), dim=1, descending=True)
            first_mask_sample = torch.mean(first_mask_sample[:, : args.T], dim=1)

            WFDRNet_AP.update(predicted_mask.flatten(), mask.flatten())
            WFDRNet_AUROC.update(predicted_mask.flatten(), mask.flatten())
            WFDRNet_Image_AUROC.update(predicted_mask_sample, mask_sample)

            First_AP.update(first_mask.flatten(), mask.flatten())
            First_AUROC.update(first_mask.flatten(), mask.flatten())
            First_Image_AUROC.update(first_mask_sample, mask_sample)

        results_mask_dir = []
        for path in dataset.mvtec_paths:
            parts = path.split("test")
            results_mask_dir.append(parts[1])

        if global_step == 0:
            all_heart_maps = [hm.cpu().numpy().squeeze() for hm in heart_maps_WFDRNet]
            global_min = min(map(np.min, all_heart_maps))
            global_max = max(map(np.max, all_heart_maps))

            visualize_and_save_images(
                args,
                results_mask_dir,
                imgs,
                mask_maps,
                heart_maps_WFDRNet,
                predicted_masks,
                "",
                category,
                global_min,
                global_max
            )

        Pixel_AP_First = First_AP.compute()
        AUROC_Pixel_First = First_AUROC.compute()
        AUROC_Image_First = First_Image_AUROC.compute()

        visualizer.add_scalar("First_Image_AUROC", AUROC_Image_First, global_step)
        visualizer.add_scalar("First_AUROC", AUROC_Pixel_First, global_step)
        visualizer.add_scalar("First_AP", Pixel_AP_First, global_step)

        Pixel_AP_WFDRNet = WFDRNet_AP.compute()
        AUROC_Pixel_WFDRNet = WFDRNet_AUROC.compute()
        AUROC_Image_WFDRNet = WFDRNet_Image_AUROC.compute()

        visualizer.add_scalar("WFDRNet_Image_AUROC", AUROC_Image_WFDRNet, global_step)
        visualizer.add_scalar("WFDRNet_AUROC", AUROC_Pixel_WFDRNet, global_step)
        visualizer.add_scalar("WFDRNet_AP", Pixel_AP_WFDRNet, global_step)

        if global_step != 0:
            print("Eval at step", global_step)
        print("================================")
        print("image_AUROC:", round(float(AUROC_Image_First), 4))
        print("pixel_AUROC:", round(float(AUROC_Pixel_First), 4))
        print("pixel_AP:", round(float(Pixel_AP_First), 4))
        print()
        print("================================")
        print("image_AUROC:", round(float(AUROC_Image_WFDRNet), 4))
        print("pixel_AUROC:", round(float(AUROC_Pixel_WFDRNet), 4))
        print("pixel_AP:", round(float(Pixel_AP_WFDRNet), 4))
        print()

        total_detail[category] = {
            "image_AUROC": round(float(AUROC_Image_WFDRNet), 4),
            "pixel_AUROC": round(float(AUROC_Pixel_WFDRNet), 4),
            "pixel_AP": round(float(Pixel_AP_WFDRNet), 4),
        }

        WFDRNet_AUROC.reset()
        WFDRNet_AP.reset()
        WFDRNet_Image_AUROC.reset()

        return total_detail[category]

def test_single_category(args, category):

    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)

    run_name = f"MVTec_test_{category}"
    run_path = os.path.join(args.log_path, run_name)
    if os.path.exists(run_path):
        shutil.rmtree(run_path)

    visualizer = SummaryWriter(log_dir=run_path)

    model = WFDRNet().cuda()
    model_file = os.path.join(args.checkpoint_path, args.base_model_name + category + ".pckl")
    assert os.path.exists(model_file), f"The model file {model_file} does not exist."

    device = f"cuda:{args.gpu_id}"
    model.load_state_dict(torch.load(model_file, map_location=device))

    evaluate_single_model(args, category, model, visualizer)

def summarize_results():
    num_categories = len(total_detail)
    if num_categories == 0:
        print("No test results found.")
        return

    sum_image_AUROC = 0
    sum_pixel_AUROC = 0
    sum_pixel_AP = 0

    for category, category_metrics in total_detail.items():
        print(f"{category}")
        print(f"  pixel_AUROC: {category_metrics['pixel_AUROC']:.4f}")
        print(f"  image_AUROC: {category_metrics['image_AUROC']:.4f}")
        print(f"  pixel_AP:  {category_metrics['pixel_AP']:.4f}")
        sum_image_AUROC += category_metrics["image_AUROC"]
        sum_pixel_AUROC += category_metrics["pixel_AUROC"]
        sum_pixel_AP += category_metrics["pixel_AP"]

    average_image_AUROC = sum_image_AUROC / num_categories
    average_pixel_AUROC = sum_pixel_AUROC / num_categories
    average_pixel_AP = sum_pixel_AP / num_categories

    print()
    print(f"Average image_AUROC: {average_image_AUROC:.4f}")
    print(f"Average pixel_AUROC: {average_pixel_AUROC:.4f}")
    print(f"Average pixel_AP:  {average_pixel_AP:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--mvtec_path", type=str, default="../datasets/mvtec/")
    parser.add_argument("--dtd_path", type=str, default="../datasets/dtd/images/")
    parser.add_argument("--checkpoint_path", type=str, default="./saved_model/")
    parser.add_argument("--base_model_name", type=str, default="Best_")
    parser.add_argument("--log_path", type=str, default="./saved_model/logs/")
    parser.add_argument("--results_mask", type=str, default="./results/")
    parser.add_argument("--mask_thresh", type=float, default=0.5)

    parser.add_argument("--bs", type=int, default=4)
    parser.add_argument("--T", type=int, default=100)

    parser.add_argument("--category", nargs="*", type=str, default=ALL_CATEGORY)
    args = parser.parse_args()

    for obj in args.category:
        assert obj in ALL_CATEGORY, f"{obj} not in the ALL_CATEGORY list"

    with torch.cuda.device(args.gpu_id):
        for obj in args.category:
            print(obj)
            test_single_category(args, obj)

    summarize_results()


if __name__ == "__main__":
    main()
