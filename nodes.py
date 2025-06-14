import os
import logging
import torch
import numpy as np
from PIL import Image
from psd_tools import PSDImage
from skimage.filters import gaussian

# 导入ComfyUI的文件夹路径模块
from comfy.utils import common_upscale
from folder_paths import get_output_directory, get_annotated_filepath
import folder_paths

# 设置日志级别
logging.basicConfig(level=logging.INFO)

# 辅助函数：将tensor转换为PIL图像
def convert_to_pil(tensor):
    # 确保tensor是4D的 [batch, height, width, channels]
    if len(tensor.shape) == 3:
        tensor = tensor.unsqueeze(0)
    
    # 转换为numpy数组
    image = tensor.squeeze().cpu().numpy()
    
    # 确保值在0-1范围内
    image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    
    # 根据通道数创建适当的PIL图像
    if image.shape[2] == 4:  # RGBA
        return Image.fromarray(image, 'RGBA')
    elif image.shape[2] == 3:  # RGB
        return Image.fromarray(image, 'RGB')
    elif image.shape[2] == 1:  # 灰度
        return Image.fromarray(image.squeeze(), 'L')
    else:
        raise ValueError(f"不支持的通道数: {image.shape[2]}")

# 辅助函数：将PIL图像转换为tensor
def convert_to_tensor(image):
    # 转换为numpy数组
    img_array = np.array(image).astype(np.float32) / 255.0
    
    # 确保是3D或4D的 [height, width, channels]
    if len(img_array.shape) == 2:  # 灰度图像
        img_array = img_array[:, :, np.newaxis]
    
    # 转换为tensor并添加batch维度
    tensor = torch.from_numpy(img_array)
    if len(tensor.shape) == 3:
        tensor = tensor.unsqueeze(0)
    
    return tensor

# 辅助函数：从RGBA图像中提取alpha蒙版
def extract_alpha_mask(tensor):
    # 确保tensor是4D的 [batch, height, width, channels]
    if len(tensor.shape) == 3:
        tensor = tensor.unsqueeze(0)
    
    # 检查是否有alpha通道
    if tensor.shape[3] != 4:
        # 如果没有alpha通道，创建一个全白蒙版
        height, width = tensor.shape[1], tensor.shape[2]
        return Image.new('L', (width, height), 255)
    
    # 提取alpha通道
    alpha = tensor[0, :, :, 3].cpu().numpy()
    
    # 转换为0-255范围的uint8
    alpha = np.clip(alpha * 255.0, 0, 255).astype(np.uint8)
    
    # 创建PIL图像
    return Image.fromarray(alpha, 'L')


# PSD数据类
class PSDData:
    def __init__(self, layers=None, width=512, height=512):
        self.layers = layers if layers is not None else []
        self.width = width
        self.height = height
    
    def create_psd(self):
        # 创建新的PSD文件
        psd = PSDImage(width=self.width, height=self.height)
        
        # 添加图层（按相反顺序添加，以便在PSD中正确显示）
        for layer_data in reversed(self.layers):
            # 创建图层
            layer = psd.create_layer(name=layer_data.get('name', 'Layer'))
            
            # 添加图像
            if 'image' in layer_data:
                pil_image = convert_to_pil(layer_data['image'])
                layer.composite(pil_image, (0, 0))
            
            # 添加蒙版（如果有）
            if 'mask' in layer_data and layer_data['mask'] is not None:
                mask_pil = convert_to_pil(layer_data['mask'])
                if mask_pil.mode != 'L':
                    mask_pil = mask_pil.convert('L')
                layer.mask.composite(mask_pil, (0, 0))
        
        return psd


# PSD图层节点
class PSDLayer:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "layer_name": ("STRING", {"default": "Layer"}),
            },
            "optional": {
                "mask": ("MASK",),
                "psd": ("PSD",),
            }
        }
    
    RETURN_TYPES = ("PSD",)
    FUNCTION = "create_layer"
    CATEGORY = "PSD"

    def create_layer(self, image, layer_name, mask=None, psd=None):
        # 获取图像尺寸
        height, width = image.shape[1], image.shape[2]
        
        # 创建或使用现有的PSD数据
        if psd is None:
            psd_data = PSDData(width=width, height=height)
        else:
            psd_data = psd
            # 更新PSD尺寸（如果需要）
            if width > psd_data.width or height > psd_data.height:
                psd_data.width = max(width, psd_data.width)
                psd_data.height = max(height, psd_data.height)
        
        # 创建图层数据
        layer_data = {
            'name': layer_name,
            'image': image
        }
        
        # 添加蒙版（如果有）
        if mask is not None:
            layer_data['mask'] = mask
        
        # 添加图层到PSD
        psd_data.layers.append(layer_data)
        
        return (psd_data,)


# 应用Alpha通道节点
class ApplyAlpha:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "blur_radius": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10.0, "step": 0.1}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_alpha_channel"
    CATEGORY = "PSD"

    def apply_alpha_channel(self, image, mask, blur_radius=0.0):
        # 确保图像是4D的 [batch, height, width, channels]
        if len(image.shape) == 3:
            image = image.unsqueeze(0)
        
        # 确保蒙版是3D的 [batch, height, width]
        if len(mask.shape) == 2:
            mask = mask.unsqueeze(0)
        
        # 获取图像尺寸
        batch_size, height, width, channels = image.shape
        
        # 创建输出图像
        if channels == 3:  # RGB转RGBA
            output = torch.zeros((batch_size, height, width, 4), dtype=image.dtype, device=image.device)
            output[:, :, :, :3] = image  # 复制RGB通道
        else:  # 已经是RGBA
            output = image.clone()
        
        # 应用蒙版到alpha通道
        for i in range(batch_size):
            # 获取当前蒙版
            current_mask = mask[i].cpu().numpy()
            
            # 应用高斯模糊（如果需要）
            if blur_radius > 0:
                current_mask = gaussian(current_mask, sigma=blur_radius)
            
            # 转换回tensor并应用到alpha通道
            alpha = torch.from_numpy(current_mask).to(image.device)
            output[i, :, :, 3] = 1.0 - alpha  # 反转蒙版值（0=透明，1=不透明）
        
        return (output,)


# 提取Alpha通道节点
class ExtractAlpha:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            }
        }
    
    RETURN_TYPES = ("MASK",)
    FUNCTION = "extract_alpha"
    CATEGORY = "PSD"

    def extract_alpha(self, image):
        # 确保图像是4D的 [batch, height, width, channels]
        if len(image.shape) == 3:
            image = image.unsqueeze(0)
        
        batch_size, height, width, channels = image.shape
        
        # 检查是否有alpha通道
        if channels != 4:
            # 如果没有alpha通道，创建一个全不透明的蒙版
            mask = torch.zeros((batch_size, height, width), dtype=torch.float32, device=image.device)
        else:
            # 提取alpha通道并反转（0=不透明，1=透明）
            mask = 1.0 - image[:, :, :, 3]
        
        return (mask,)


# 保存PSD节点
class SavePSD:
    def __init__(self):
        self.output_dir = get_output_directory()
        self.type = "output"
        self.prefix_append = ""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "filename_prefix": ("STRING", {"default": "ComfyUI"}),
                "file_mode": (["single_file", "multi_file"], {"default": "single_file"}),
                "alpha_name": ("STRING", {"default": "_mask_"}),
                "alpha_name_mode": (["simple", "layer_name"], {"default": "simple"}),
            },
            "optional": {
                "psd": ("PSD",),
                "save_path": ("STRING", {"default": ""}),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }
    
    RETURN_TYPES = ()
    FUNCTION = "save_psd"
    OUTPUT_NODE = True
    CATEGORY = "PSD"

    def save_psd(self, images, filename_prefix, file_mode="single_file", alpha_name="_mask_", alpha_name_mode="simple", psd=None, save_path="", prompt=None, extra_pnginfo=None):
        try:
            # 检查是否有图像或PSD对象
            if images is not None or psd is not None:
                # 确定输出文件夹
                custom_output_folder = save_path.strip() != ""
                if custom_output_folder:
                    full_output_folder = save_path
                    os.makedirs(full_output_folder, exist_ok=True)
                else:
                    full_output_folder = self.output_dir
                
                # 获取计数器
                counter = len(os.listdir(full_output_folder))
                
                # 获取文件名
                filename = filename_prefix
                if filename.endswith(".psd"):
                    filename = filename[:-4]
                
                # 如果有PSD对象，使用它
                if psd is not None:
                    # 创建PSD并保存
                    psd_image = psd.create_psd()
                    file = f"{filename}_{counter:05}_.psd"
                    file_path = os.path.join(full_output_folder, file)
                    psd_image.save(file_path)
                    logging.info(f"PSD文件已成功保存: {file_path}")
                
                # 如果有图像，创建新的PSD
                elif images is not None:
                    # 确保图像是4D的 [batch, height, width, channels]
                    if len(images.shape) == 3:
                        images = images.unsqueeze(0)
                    
                    # 获取图像尺寸
                    batch_size, height, width, channels = images.shape
                    
                    if file_mode == "single_file":
                        # 创建单个PSD文件包含所有图像作为图层
                        new_psd = PSDImage(width=width, height=height)
                        
                        # 添加图层（按相反顺序添加，以便在PSD中正确显示）
                        for i in range(batch_size - 1, -1, -1):
                            # 转换为PIL图像
                            pil_image = convert_to_pil(images[i])
                            
                            # 创建图层
                            layer_name = f"Layer {i + 1}"
                            layer = new_psd.create_layer(name=layer_name)
                            layer.composite(pil_image, (0, 0))
                            
                            # 如果有alpha通道，添加为图层蒙版
                            if channels == 4:
                                alpha_mask = extract_alpha_mask(images[i])
                                layer.mask.composite(alpha_mask, (0, 0))
                        
                        # 生成文件名
                        file = f"{filename_prefix}_{counter:05}_.psd"
                        file_path = os.path.join(full_output_folder, file)
                        
                        # 保存PSD
                        new_psd.save(file_path)
                        logging.info(f"PSD文件已成功保存: {file_path}")
                        
                    elif file_mode == "multi_file":
                        # 为每个图像创建一个PSD文件
                        for i in range(batch_size):
                            # 创建新的PSD
                            new_psd = PSDImage(width=width, height=height)
                            
                            # 转换为PIL图像
                            pil_image = convert_to_pil(images[i])
                            
                            # 创建图层
                            layer_name = "Layer 1"
                            layer = new_psd.create_layer(name=layer_name)
                            layer.composite(pil_image, (0, 0))
                            
                            # 如果有alpha通道，添加为图层蒙版
                            if channels == 4:
                                alpha_mask = extract_alpha_mask(images[i])
                                layer.mask.composite(alpha_mask, (0, 0))
                            
                            # 生成文件名
                            if custom_output_folder:
                                # 使用自定义路径的简单文件名
                                file = f"{filename_prefix}_{counter + i:05}_.psd"
                                file_path = os.path.join(full_output_folder, file)
                            else:
                                # 使用默认路径的文件名
                                file = f"{filename.replace('%batch_num%', str(i))}_{counter + i:05}_.psd"
                                file_path = os.path.join(full_output_folder, file)
                            
                            # 保存PSD
                            new_psd.save(file_path)
                            logging.info(f"PSD文件已成功保存: {file_path}")

                # 替代方案: 如果PSD保存失败，尝试保存为PNG
                if images is not None and 'e' in locals():
                    for i, img_tensor in enumerate(images):
                        try:
                            img_pil = convert_to_pil(img_tensor)
                            alt_file = f"{filename.replace('%batch_num%', str(i))}_{counter:05}_.png"
                            alt_path = os.path.join(full_output_folder, alt_file)
                            img_pil.save(alt_path)
                        except Exception as alt_e:
                            logging.warning(f"保存为PNG也失败了: {str(alt_e)}")
            else:
                logging.warning("没有提供PSD对象或图像，无法保存PSD文件")
                
        except Exception as e:
            logging.warning(f"保存PSD时发生错误: {str(e)}")

        # 返回空元组
        return ()


class ConvertPSDToImage:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "psd": ("PSD",),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "preview"
    CATEGORY = "PSD"

    def preview(self, psd):
        # 创建PSD并合成为图像
        img = psd.create_psd().composite()
        # 转换为tensor
        tensor = convert_to_tensor(img)
        return (tensor,)


class PSD2PNG:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "psd": ("PSD",),
                "layer_index": ("INT", {"default": 0, "min": 0, "max": 999, "step": 1}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "MASK", "FLOAT")
    RETURN_NAMES = ("image", "top_layer", "bottom_layer", "mask", "layer_exists")
    FUNCTION = "process_psd"
    CATEGORY = "PSD"

    def get_image_and_mask(self, psd_image, layer_list, layer_number):
        mask_out = None
        layer_image = layer_list[layer_number].composite()
        layer_obj = layer_list[layer_number]
        # 创建一个空白画布
        canvas_size = (int(psd_image.width), int(psd_image.height))
        canvas_image_obj = Image.new('RGBA', canvas_size, (0, 0, 0, 0))
        # 将图层复制到新画布上
        layer_bbox = layer_obj.bbox
        offset = (layer_bbox[0], layer_bbox[1])
        canvas_image_obj.paste(layer_image, offset)
        image_out = canvas_image_obj.convert("RGBA")
        image_out = np.array(image_out).astype(np.float32) / 255.0
        image_out = torch.from_numpy(image_out)[None,]
        if 'A' in canvas_image_obj.getbands():
            mask = np.array(canvas_image_obj.getchannel('A')).astype(np.float32) / 255.0
            mask = 1. - torch.from_numpy(mask)
        else:
            mask = torch.zeros((64, 64), dtype=torch.float32, device="cpu")
        mask_out = mask.unsqueeze(0)
        return image_out, mask_out

    def process_psd(self, psd, layer_index):
        # 使用PSD对象创建PSD文件
        psd_image = psd.create_psd()
        
        # 获取输入图像
        input_image = psd_image.composite()
        input_image = np.array(input_image).astype(np.float32) / 255.0
        input_image = torch.from_numpy(input_image)[None,]
        
        top_image = None
        bottom_image = None
        mask_out = None
        is_exist_layer = 1.0
        
        from psd_tools.api.layers import Layer
        layer_list = [layer for layer in psd_image.descendants() if isinstance(layer, Layer)]
        
        if not layer_list:
            # 如果没有图层，返回整个图像
            return (input_image, input_image, input_image, 
                    torch.zeros((1, 64, 64), dtype=torch.float32, device="cpu"), 0.0)
        
        top_layer_number = len(layer_list) - 1
        
        if layer_index == 0:
            # 返回整个PSD图像
            image_out = input_image
            mask_out = torch.zeros((1, 64, 64), dtype=torch.float32, device="cpu")
            if top_layer_number >= 0:
                top_image = self.get_image_and_mask(psd_image, layer_list, top_layer_number)[0]
                bottom_image = self.get_image_and_mask(psd_image, layer_list, 0)[0]
            else:
                top_image = bottom_image = input_image
        elif len(layer_list) == 1:
            # 只有一个图层
            top_image, mask_out = self.get_image_and_mask(psd_image, layer_list, 0)
            bottom_image = top_image
            image_out = top_image
        elif len(layer_list) > 1:
            # 多个图层
            top_image, mask_out = self.get_image_and_mask(psd_image, layer_list, top_layer_number)
            bottom_image = self.get_image_and_mask(psd_image, layer_list, 0)[0]
            
            if (layer_index - 1) > top_layer_number:
                # 请求的图层索引超出范围
                image_out = top_image
                is_exist_layer = 0.0
            else:
                # 返回请求的图层
                image_out, mask_out = self.get_image_and_mask(psd_image, layer_list, layer_index - 1)
        else:
            # 非PSD文件处理
            image_out = input_image
            top_image = bottom_image = input_image
            mask = torch.zeros((64, 64), dtype=torch.float32, device="cpu")
            mask_out = mask.unsqueeze(0)
            
        return (image_out, top_image, bottom_image, mask_out, is_exist_layer)


class PSD2PNGFromFile:
    def __init__(self):
        self.input_dir = folder_paths.get_input_directory()

    @classmethod
    def INPUT_TYPES(cls):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f)) and f.lower().endswith('.psd')]
        if len(files) == 0:
            files = ["empty"]
        return {
            "required": {
                "psd_file": (sorted(files), {"image_upload": True}),
                "layer_index": ("INT", {"default": 0, "min": 0, "max": 999, "step": 1}),
            },
            "optional": {
                "psd_path": ("STRING", {"default": ""}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "MASK", "FLOAT")
    RETURN_NAMES = ("image", "top_layer", "bottom_layer", "mask", "layer_exists")
    FUNCTION = "process_psd"
    CATEGORY = "PSD"

    def get_image_and_mask(self, psd_image, layer_list, layer_number):
        mask_out = None
        layer_image = layer_list[layer_number].composite()
        layer_obj = layer_list[layer_number]
        # 创建一个空白画布
        canvas_size = (int(psd_image.width), int(psd_image.height))
        canvas_image_obj = Image.new('RGBA', canvas_size, (0, 0, 0, 0))
        # 将图层复制到新画布上
        layer_bbox = layer_obj.bbox
        offset = (layer_bbox[0], layer_bbox[1])
        canvas_image_obj.paste(layer_image, offset)
        image_out = canvas_image_obj.convert("RGBA")
        image_out = np.array(image_out).astype(np.float32) / 255.0
        image_out = torch.from_numpy(image_out)[None,]
        if 'A' in canvas_image_obj.getbands():
            mask = np.array(canvas_image_obj.getchannel('A')).astype(np.float32) / 255.0
            mask = 1. - torch.from_numpy(mask)
        else:
            mask = torch.zeros((64, 64), dtype=torch.float32, device="cpu")
        mask_out = mask.unsqueeze(0)
        return image_out, mask_out

    def process_psd(self, psd_file, layer_index, psd_path=""):
        from pathlib import Path
        from psd_tools.api.layers import Layer

        # 确定PSD文件路径
        if psd_file == "empty" and not psd_path:
            # 创建一个空白图像
            empty_image = torch.zeros((1, 512, 512, 3), dtype=torch.float32, device="cpu")
            empty_mask = torch.zeros((1, 512, 512), dtype=torch.float32, device="cpu")
            logging.warning("没有提供有效的PSD文件，请上传PSD文件或提供有效的文件路径")
            return (empty_image, empty_image, empty_image, empty_mask, 0.0)
        
        if psd_path and os.path.exists(psd_path):
            file_path = Path(psd_path)
        else:
            if psd_file == "empty":
                # 创建一个空白图像
                empty_image = torch.zeros((1, 512, 512, 3), dtype=torch.float32, device="cpu")
                empty_mask = torch.zeros((1, 512, 512), dtype=torch.float32, device="cpu")
                logging.warning("没有提供有效的PSD文件，请上传PSD文件或提供有效的文件路径")
                return (empty_image, empty_image, empty_image, empty_mask, 0.0)
            file_path = Path(folder_paths.get_annotated_filepath(psd_file))
        
        # 打开PSD文件
        try:
            i = Image.open(file_path)
            input_image = i.convert("RGB")
            input_image = np.array(input_image).astype(np.float32) / 255.0
            input_image = torch.from_numpy(input_image)[None,]
        except Exception as e:
            logging.warning(f"打开PSD文件时出错: {str(e)}")
            # 创建一个空白图像
            empty_image = torch.zeros((1, 512, 512, 3), dtype=torch.float32, device="cpu")
            empty_mask = torch.zeros((1, 512, 512), dtype=torch.float32, device="cpu")
            return (empty_image, empty_image, empty_image, empty_mask, 0.0)
        
        top_image = None
        bottom_image = None
        mask_out = None
        is_exist_layer = 1.0
        
        if file_path.suffix.lower() == ".psd":  
            psd_image = PSDImage.open(file_path)
            layer_list = [layer for layer in psd_image.descendants() if isinstance(layer, Layer)]
            
            if not layer_list:
                # 如果没有图层，返回整个图像
                return (input_image, input_image, input_image, 
                        torch.zeros((1, 64, 64), dtype=torch.float32, device="cpu"), 0.0)
            
            top_layer_number = len(layer_list) - 1
            
            if layer_index == 0:
                # 返回整个PSD图像
                image_out = input_image
                mask_out = torch.zeros((1, 64, 64), dtype=torch.float32, device="cpu")
                if top_layer_number >= 0:
                    top_image = self.get_image_and_mask(psd_image, layer_list, top_layer_number)[0]
                    bottom_image = self.get_image_and_mask(psd_image, layer_list, 0)[0]
                else:
                    top_image = bottom_image = input_image
            elif len(layer_list) == 1:
                # 只有一个图层
                top_image, mask_out = self.get_image_and_mask(psd_image, layer_list, 0)
                bottom_image = top_image
                image_out = top_image
            elif len(layer_list) > 1:
                # 多个图层
                top_image, mask_out = self.get_image_and_mask(psd_image, layer_list, top_layer_number)
                bottom_image = self.get_image_and_mask(psd_image, layer_list, 0)[0]
                
                if (layer_index - 1) > top_layer_number:
                    # 请求的图层索引超出范围
                    image_out = top_image
                    is_exist_layer = 0.0
                else:
                    # 返回请求的图层
                    image_out, mask_out = self.get_image_and_mask(psd_image, layer_list, layer_index - 1)
        else:
            # 非PSD文件处理
            if 'A' in i.getbands():
                mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
                mask = 1. - torch.from_numpy(mask)
            else:
                mask = torch.zeros((64, 64), dtype=torch.float32, device="cpu")
            
            image_out = input_image
            top_image = bottom_image = input_image
            mask_out = mask.unsqueeze(0)

        return (image_out, top_image, bottom_image, mask_out, is_exist_layer)
    
    @classmethod
    def IS_CHANGED(cls, psd_file, layer_index, psd_path=""):
        import hashlib
        from pathlib import Path
        
        # 确定PSD文件路径
        if psd_path and os.path.exists(psd_path):
            file_path = Path(psd_path)
        else:
            file_path = Path(folder_paths.get_annotated_filepath(psd_file))
        
        # 计算文件哈希值
        m = hashlib.sha256()
        with open(file_path, 'rb') as f:
            m.update(f.read())
        
        # 返回哈希值和图层索引的组合
        return f"{m.digest().hex()}_{layer_index}"

# 注册节点
NODE_CLASS_MAPPINGS = {
    "PSD Layer": PSDLayer,
    "Apply Alpha": ApplyAlpha,
    "Extract Alpha": ExtractAlpha,
    "Save PSD": SavePSD,
    "Convert PSD To Image": ConvertPSDToImage,
    "PSD To PNG": PSD2PNG,
    "PSD To PNG From File": PSD2PNGFromFile
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PSD Layer": "comfyui-psd: PSD Layer",
    "Apply Alpha": "comfyui-psd: Apply Alpha",
    "Extract Alpha": "comfyui-psd: Extract Alpha",
    "Save PSD": "comfyui-psd: Save PSD",
    "Convert PSD To Image": "comfyui-psd: Convert PSD To Image",
    "PSD To PNG": "comfyui-psd: PSD To PNG",
    "PSD To PNG From File": "comfyui-psd: PSD To PNG From File"
}