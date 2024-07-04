import os
import numpy as np
import pandas as pd
import torch
from monai.visualize import GradCAMpp, OcclusionSensitivity
from .data_generator import load_dataset, load_efficientnet_b0_imaging_model
import matplotlib.pyplot as plt
import tqdm

def image_show(data, label, nims, the_slice):
    nims = len(data)
    if nims < 6:
        shape = (1, nims)
    else:
        shape = int(np.floor(np.sqrt(nims))), int(np.ceil(np.sqrt(nims)))
    fig, axes = plt.subplots(*shape, figsize=(5, 5), facecolor="white")
    axes = np.asarray(axes) if nims == 1 else axes
    for d, ax in zip(data, axes.ravel()):
        # channel last for matplotlib
        im =  torch.from_numpy(data).to('cuda').detach().cpu().numpy()[0, 0, the_slice, ...]
        ax.imshow(im, cmap="gray")
        ax.set_title(label, fontsize=25)
        ax.axis("off")
    plt.show()

def saliency(model, d, label, gradcampp, occ_sens):
    ims = []
    titles = []
    log_scales = []

    img = torch.from_numpy(d).to('cuda')
    pred_logits = model(img)
    pred_label = pred_logits.argmax(dim=1).item()
    pred_prob = int(torch.sigmoid(pred_logits)[0, 1].item() * 100)
    print(pred_logits, pred_label, pred_prob)
    # Image
    ims.append(img[0])
    title = f"GT: {label}, "
    title += f"pred: {pred_label} ({pred_prob}%)"
    titles.append(title)
    log_scales.append(False)

    # Occlusion sensitivity images
    occ_map, _ = occ_sens(img)
    ims.append(occ_map[0, pred_label][None])
    titles.append("Occ. sens.")
    log_scales.append(False)

    # GradCAM
    res_cam_pp = gradcampp(x=img, class_idx=pred_label)[0]
    ims.append(res_cam_pp)
    titles.append("GradCAMpp")
    log_scales.append(False)

    return ims, titles, log_scales

def add_im(im, title, log_scale, row, col, num_examples, cmap, axes, fig):
    ax = axes[row, col] if num_examples > 1 else axes[col]
    if isinstance(im, torch.Tensor):
        im = im.detach().cpu()
    the_slice = int(im[1]/2)
    im_show = ax.imshow(im[0, the_slice, :, :], cmap=cmap)
    ax.set_title(title, fontsize=25)
    ax.axis("off")
    if col > 0:
        fig.colorbar(im_show, ax=ax)


def add_row(ims, titles, log_scales, row, axes, num_examples, fig):
    for col, (im, title, log_scale) in enumerate(zip(ims, titles, log_scales)):
        cmap = "gray" if col == 0 else "jet"
        if log_scale and im.min() < 0:
            im -= im.min()
        add_im(im, title, log_scale, row, col, num_examples, cmap, axes, fig)


def create_gradcam(args):
    imaging_models = args.optimal_imaging_model_paths
    print(f'Using the following paths:{imaging_models}')
    for fold in range(args.num_folds):
        for key, val in imaging_models.items():
            print(key, val)
            model_path = val
            print(model_path)
            model = load_efficientnet_b0_imaging_model(model_path, head=True)
            model.eval()

            all_valid_images, all_valid_labels, all_valid_ids = load_dataset(
                args.overview_dir + 'overview_testset.xlsx')
            
            df = pd.read_excel(args.overview_dir + 'overview_testset.xlsx')
            rand_idxs = df.index[df['lbls'] == 1].tolist()
            print(rand_idxs)

            for idx in tqdm.tqdm(range(len(rand_idxs))):
                # for name, _ in model.named_modules(): print(name)
                target_layer = "_conv_head"
                gradcampp = GradCAMpp(model, target_layers=target_layer)
                occ_sens = OcclusionSensitivity(
                    model,
                    mask_size=32,
                    n_batch=1,
                    overlap=0.75,
                    verbose=False,
                )

                ims, titles, log_scales = saliency(model, all_valid_images[idx], all_valid_labels[idx], gradcampp, occ_sens)
                num_cols = len(ims)
                subplot_shape = [1, num_cols]
                figsize = [i * 5 for i in subplot_shape][::-1]
                fig, axes = plt.subplots(*subplot_shape, figsize=figsize, facecolor="white")
                add_row(ims, titles, log_scales, 1, axes, 1, fig)
                plt.tight_layout()
                path = model_path.split('.pt')[0]
                plt.savefig(args.results_dir + 'images/' + path + '_' + all_valid_ids[idx] + '.png')


