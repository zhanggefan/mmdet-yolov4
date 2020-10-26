from .. import PIPELINES
import mmcv
import numpy as np


@PIPELINES.register_module()
class RoundPad(object):
    """Pad the image & mask.

    There are two padding modes: (1) pad to a fixed size and (2) pad to the
    minimum size that is divisible by some number.
    Added keys are "pad_shape", "pad_fixed_size", "pad_size_divisor",

    Args:
        size (tuple, optional): Fixed padding size.
        size_divisor (int, optional): The divisor of padded size.
        pad_val (float, optional): Padding value, 0 by default.
    """

    def __init__(self, size=None, size_divisor=None, pad_val=0):
        self.size = size
        self.size_divisor = size_divisor
        self.pad_val = pad_val
        # only one of size and size_divisor should be valid
        assert size is not None or size_divisor is not None
        assert size is None or size_divisor is None

    def __call__(self, results):
        """Call function to pad images, masks, semantic segmentation maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Updated result dict.
        """
        """Pad images according to ``self.size``."""
        for key in results.get('img_fields', ['img']):
            img = results[key]
            ori_h, ori_w = img.shape[:2]
            if self.size is not None:
                pad_h, pad_w = self.size
            elif self.size_divisor is not None:
                divisor = self.size_divisor
                pad_h = int(np.ceil(ori_h / divisor)) * divisor
                pad_w = int(np.ceil(ori_w / divisor)) * divisor

            pad_top = (pad_h - ori_h) // 2
            pad_bottom = pad_h - ori_h - pad_top
            pad_left = (pad_w - ori_w) // 2
            par_right = pad_w - ori_w - pad_left
            padded_img = mmcv.impad(
                results[key], padding=(pad_left, pad_top, par_right, pad_bottom), pad_val=self.pad_val)
            results[key] = padded_img

        results['pad_shape'] = padded_img.shape
        results['pad_fixed_size'] = self.size
        results['pad_size_divisor'] = self.size_divisor

        # crop bboxes accordingly and clip to the image boundary
        for key in results.get('bbox_fields', []):
            # e.g. gt_bboxes and gt_bboxes_ignore
            bbox_offset = np.array([pad_left, pad_top, pad_left, pad_top],
                                   dtype=np.float32)
            results[key] += bbox_offset
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(size={self.size}, '
        repr_str += f'size_divisor={self.size_divisor}, '
        repr_str += f'pad_val={self.pad_val})'
        return repr_str
