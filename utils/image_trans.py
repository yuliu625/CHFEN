from PIL import Image, ImageDraw, ImageFont


def trans_np_array_to_image(np_array_image):
    """将numpy array的图片转换为原始的图片。"""
    pil_img = Image.fromarray(np_array_image)
    return pil_img
