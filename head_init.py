"""
head_init.py — Final layer initialization (student-implemented).

Students: Implement `init_last_layer` to control how the new classification
head is initialized before fine-tuning begins. The skeleton below uses
Kaiming uniform weights and zero bias — you are expected to experiment with
alternatives (e.g. Xavier, orthogonal, small-scale random, learned bias init).
"""

import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights


def init_last_layer(layer: nn.Linear) -> None:
    """Initialize the weights and bias of the final classification layer in-place.

    This function is called once during model construction (see model.py).
    Modify it to experiment with different initialization strategies and observe
    their effect on the "initialized head" evaluation checkpoint.

    Args:
        layer: The ``nn.Linear`` layer that serves as the new CIFAR100 head.
               Modifies the layer in-place; return value is ignored.

    Student task:
        Replace or extend the skeleton below. Some strategies to consider:
          - ``nn.init.xavier_uniform_``  — preserves variance across layers
          - ``nn.init.orthogonal_``      — encourages diverse feature directions
          - Small-scale init (e.g. scale weights by 0.01) — conservative start
          - Non-zero bias init           — useful when class priors are known
    """
    # Use the pretrained ImageNet classifier as a semantic prior for the new
    # CIFAR100 head.  CIFAR100 contains many classes that overlap with, or are
    # close to, ImageNet categories; copying/averaging those rows gives the
    # zero-order optimizer a far better starting point than random logits.
    imagenet_rows = [
        [948, 956],
        [1, 393, 396, 397],
        [520, 431, 850],
        [294, 295, 296, 297],
        [337],
        [564, 520, 831],
        [309],
        [300, 302, 303, 304, 305, 306, 307],
        [444, 671, 870],
        [440, 737, 898, 907, 720],
        [659, 809],
        [981, 982, 610, 608],
        [821, 839, 888],
        [779, 874, 654, 656],
        [321, 322, 323, 324, 325, 326],
        [354],
        [653, 473, 412],
        [483, 497, 698],
        [79, 313, 315],
        [345, 346, 347, 351],
        [423, 559, 765],
        [367, 365, 366],
        [409, 530, 892, 604],
        [405, 417, 970, 979],
        [314],
        [831, 765, 532],
        [118, 119, 120, 121, 125],
        [49, 50],
        [968, 504, 647, 572],
        [51],
        [147, 148, 149, 150, 389],
        [385, 386, 101],
        [0, 389, 391, 394],
        [970, 975, 976, 979],
        [277, 278, 279, 280],
        [981, 982, 578, 655, 614],
        [333, 338],
        [660, 449, 448, 580, 425],
        [104],
        [508, 878, 810],
        [619, 846, 470],
        [621],
        [288, 289, 290, 293],
        [291],
        [38, 40, 41, 42, 43, 44, 45, 46, 47, 48],
        [122, 123],
        [982, 834, 610, 400],
        [988, 990, 984],
        [670, 665],
        [970, 972, 976, 979],
        [673, 674],
        [947, 991, 992, 993, 994, 995, 996, 997],
        [988, 990],
        [950],
        [986, 985],
        [360],
        [978, 975, 980],
        [948, 949, 950, 951, 952, 953, 954, 956, 957],
        [717, 864, 867],
        [958, 970, 976],
        [970, 979, 976],
        [923, 729],
        [985, 984],
        [334],
        [102, 105, 106, 338, 336],
        [330, 331, 332],
        [362, 298, 299],
        [5, 6],
        [919, 920, 705, 751],
        [657, 744, 812],
        [985, 883],
        [978, 975, 973, 801],
        [150, 149],
        [2, 3, 4],
        [356, 357, 358, 359],
        [361],
        [538, 832, 682, 497],
        [113, 114],
        [52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68],
        [72, 73, 74, 75, 76, 77],
        [335, 382, 336],
        [829, 874, 547],
        [985, 984],
        [945, 943, 939],
        [532, 526, 736],
        [847, 586],
        [487, 528, 707],
        [851, 598, 664],
        [292, 282],
        [866, 595, 730],
        [466, 547, 565],
        [0, 391, 389],
        [986, 985],
        [33, 34, 35, 36, 37],
        [894, 493, 492],
        [147, 148],
        [988, 990, 984],
        [269, 270, 271, 272, 275],
        [578, 655, 614, 834],
        [110, 111],
    ]

    state = ResNet18_Weights.IMAGENET1K_V1.get_state_dict(progress=False)
    source_weight = state["fc.weight"]
    source_bias = state["fc.bias"]

    with torch.no_grad():
        nn.init.xavier_uniform_(layer.weight, gain=0.25)
        nn.init.zeros_(layer.bias)

        for class_idx, rows in enumerate(imagenet_rows):
            selected = source_weight[rows]
            row = selected.mean(dim=0)
            target_norm = selected.norm(dim=1).mean()
            row_norm = row.norm().clamp_min(1e-12)
            layer.weight[class_idx].copy_(row * (target_norm / row_norm))
            layer.bias[class_idx].copy_(source_bias[rows].mean())

        layer.bias.sub_(layer.bias.mean())
