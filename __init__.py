"""ComfyUI PSD

提供强大的PSD文件处理能力。

@author:ababxg
@title: comfyui-psd
@description: 整合了多种PSD处理功能的ComfyUI插件
"""

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
