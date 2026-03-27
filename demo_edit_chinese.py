"""
Demo script: Edit text in an existing image using AnyText2.

This script takes a demo image (example_images/ref3.jpg - a yellow sign with Chinese text)
and replaces the text with "中文编辑测试" using AnyText2's text editing mode.

Usage:
    # Basic usage (requires models downloaded to ./models/)
    python demo_edit_chinese.py

    # Use FP32 (for GPUs without FP16 support)
    python demo_edit_chinese.py --use_fp32

    # Disable Chinese->English translator to save ~4GB VRAM
    python demo_edit_chinese.py --no_translator

    # Custom input image with position mask
    python demo_edit_chinese.py --ori_image path/to/image.jpg --pos_image path/to/mask.png

Prerequisites:
    1. Install environment: conda env create -f environment.yaml && conda activate anytext2
    2. Download model weights:
       python -c "from modelscope import snapshot_download; snapshot_download('iic/cv_anytext2')"
       Then move weights to ./models/
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import sys
import argparse
import cv2
import numpy as np
from PIL import Image

from ms_wrapper import AnyText2Model
from util import check_channels, resize_image, save_images


def parse_args():
    parser = argparse.ArgumentParser(
        description='Demo: Edit text in an image to "中文编辑测试" using AnyText2'
    )
    parser.add_argument(
        '--ori_image',
        type=str,
        default='example_images/ref3.jpg',
        help='Path to the original image to edit'
    )
    parser.add_argument(
        '--pos_image',
        type=str,
        default='example_images/edit3.png',
        help='Path to the position mask image (black regions = areas to edit)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='demo_output',
        help='Directory to save output images'
    )
    parser.add_argument(
        '--text',
        type=str,
        default='中文编辑测试',
        help='Text to write into the image'
    )
    parser.add_argument(
        '--img_prompt',
        type=str,
        default='一个黄色标志牌',
        help='Image description prompt (describes the scene)'
    )
    parser.add_argument(
        '--use_fp32',
        action='store_true',
        default=False,
        help='Use FP32 instead of FP16 during inference'
    )
    parser.add_argument(
        '--no_translator',
        action='store_true',
        default=False,
        help='Disable the Chinese->English translator (saves ~4GB VRAM)'
    )
    parser.add_argument(
        '--model_path',
        type=str,
        default='models/anytext_v2.0.ckpt',
        help='Path to the AnyText2 model checkpoint'
    )
    parser.add_argument(
        '--font_path',
        type=str,
        default='font/Arial_Unicode.ttf',
        help='Path to font file for text rendering'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=68988613,
        help='Random seed for reproducibility (-1 for random)'
    )
    parser.add_argument(
        '--num_images',
        type=int,
        default=2,
        help='Number of images to generate'
    )
    parser.add_argument(
        '--ddim_steps',
        type=int,
        default=20,
        help='Number of DDIM sampling steps'
    )
    return parser.parse_args()


def create_position_mask(ori_image, pos_image):
    """
    Create a position mask for text editing.

    The pos_image (edit*.png) has black regions where text should be edited.
    We invert it to get white regions = text positions on a dark background,
    then resize to match the original image dimensions.

    Args:
        ori_image: Original image (numpy array, RGB, HWC)
        pos_image: Position mask image (numpy array, RGB, HWC) where
                   dark/black regions indicate areas to edit

    Returns:
        pos_imgs: Position mask (numpy array) with white regions for text areas
    """
    edit_image = ori_image.clip(1, 255)
    edit_image = check_channels(edit_image)
    edit_image = resize_image(edit_image, max_length=1024)
    h, w = edit_image.shape[:2]

    # Invert: dark regions in pos_image become white in pos_imgs
    pos_imgs = 255 - pos_image
    pos_imgs = cv2.resize(pos_imgs, (w, h))
    return pos_imgs, edit_image


def build_text_prompt(text_lines):
    """
    Build text prompt with quoted text lines.

    AnyText2 expects text wrapped in double quotes, e.g.:
        '"Hello" "World"' for two lines of text

    Args:
        text_lines: list of text strings, one per line

    Returns:
        Formatted text prompt string
    """
    quoted = [f'"{t}"' for t in text_lines]
    return ' '.join(quoted)


def main():
    args = parse_args()

    # Validate input files
    if not os.path.exists(args.ori_image):
        print(f'Error: Original image not found: {args.ori_image}')
        sys.exit(1)
    if not os.path.exists(args.pos_image):
        print(f'Error: Position mask image not found: {args.pos_image}')
        sys.exit(1)
    if not os.path.exists(args.model_path):
        print(f'Error: Model checkpoint not found: {args.model_path}')
        print('Please download model weights first:')
        print('  python -c "from modelscope import snapshot_download; snapshot_download(\'iic/cv_anytext2\')"')
        print('  Then move the downloaded files to ./models/')
        sys.exit(1)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load the model
    print('Loading AnyText2 model...')
    infer_params = {
        'use_fp16': not args.use_fp32,
        'use_translator': not args.no_translator,
        'font_path': args.font_path,
        'model_path': args.model_path,
    }
    inference = AnyText2Model(model_dir='./models', **infer_params).cuda(0)
    print('Model loaded successfully.')

    # Read input images
    ori_image = cv2.imread(args.ori_image)
    assert ori_image is not None, f'Failed to read original image: {args.ori_image}'
    ori_image = ori_image[..., ::-1]  # BGR -> RGB

    pos_image = cv2.imread(args.pos_image)
    assert pos_image is not None, f'Failed to read position mask: {args.pos_image}'
    pos_image = pos_image[..., ::-1]  # BGR -> RGB

    # Prepare the position mask
    pos_imgs, edit_image = create_position_mask(ori_image, pos_image)

    # Build prompts
    # Split text into lines if it contains newlines; otherwise single line
    text_lines = args.text.split('\\n') if '\\n' in args.text else [args.text]
    text_prompt = build_text_prompt(text_lines)
    img_prompt = args.img_prompt

    print(f'Image prompt: {img_prompt}')
    print(f'Text prompt: {text_prompt}')
    print(f'Editing text to: {args.text}')
    print(f'Generating {args.num_images} image(s)...')

    # Run inference in text-editing mode
    input_data = {
        'img_prompt': img_prompt,
        'text_prompt': text_prompt,
        'seed': args.seed,
        'draw_pos': pos_imgs,
        'ori_image': ori_image,
    }
    params = {
        'mode': 'text-editing',
        'sort_priority': '↕',
        'show_debug': False,
        'revise_pos': False,
        'image_count': args.num_images,
        'ddim_steps': args.ddim_steps,
        'image_width': 512,
        'image_height': 512,
        'strength': 1.0,
        'attnx_scale': 1.0,
        'font_hollow': True,
        'cfg_scale': 7.5,
        'eta': 0.0,
        'a_prompt': 'best quality, extremely detailed, 4k, HD, supper legible text, clear text edges, clear strokes, neat writing, no watermarks',
        'n_prompt': 'low-res, bad anatomy, extra digit, fewer digits, cropped, worst quality, low quality, watermark, unreadable text, messy words, distorted text, disorganized writing, advertising picture',
        'glyline_font_path': '',
        'font_hint_image': [None] * 5,
        'font_hint_mask': [None] * 5,
        'text_colors': ' '.join(['500,500,500'] * 5),
    }

    results, rtn_code, rtn_warning, debug_info = inference(input_data, **params)

    if rtn_code < 0:
        print(f'Error: {rtn_warning}')
        sys.exit(1)

    if rtn_warning:
        print(f'Warning: {rtn_warning}')

    # Save results
    save_images(results, args.output_dir)
    for idx, img in enumerate(results):
        out_path = os.path.join(args.output_dir, f'edited_{idx}.png')
        cv2.imwrite(out_path, img[..., ::-1])  # RGB -> BGR for cv2
        print(f'Saved: {out_path}')

    print(f'Done! {len(results)} image(s) saved to {args.output_dir}/')


if __name__ == '__main__':
    main()
