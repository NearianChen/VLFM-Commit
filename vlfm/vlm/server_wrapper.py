# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

import base64
import os
import random
import socket
import time
from typing import Any, Dict

import cv2
import numpy as np
import requests
from flask import Flask, jsonify, request


class ServerMixin:
    """
    服务器混入类，为模型服务提供基本功能。
    子类需要实现process_payload方法来处理请求负载。
    """
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    def process_payload(self, payload: dict) -> dict:
        """
        处理请求负载的方法，由子类实现。
        
        Args:
            payload: 包含请求数据的字典
            
        Returns:
            处理结果的字典
        """
        raise NotImplementedError


def host_model(model: Any, name: str, port: int = 5000) -> None:
    """
    将模型作为REST API托管使用Flask框架。
    
    Args:
        model: 要托管的模型实例
        name: API端点名称
        port: 服务器端口号，默认为5000
    """
    app = Flask(__name__)

    @app.route(f"/{name}", methods=["POST"])
    def process_request() -> Dict[str, Any]:
        """处理HTTP POST请求并返回模型处理结果"""
        payload = request.json
        return jsonify(model.process_payload(payload))

    app.run(host="localhost", port=port)


def bool_arr_to_str(arr: np.ndarray) -> str:
    """
    将布尔数组转换为字符串以便传输。
    
    Args:
        arr: 布尔类型的NumPy数组
        
    Returns:
        base64编码的字符串
    """
    packed_str = base64.b64encode(arr.tobytes()).decode()
    return packed_str


def str_to_bool_arr(s: str, shape: tuple) -> np.ndarray:
    """
    将字符串转换回布尔数组。
    
    Args:
        s: base64编码的字符串
        shape: 数组的目标形状
        
    Returns:
        还原后的布尔数组
    """
    # 使用base64解码将字符串转回字节
    bytes_ = base64.b64decode(s)

    # 将字节转换为np.uint8数组
    bytes_array = np.frombuffer(bytes_, dtype=np.uint8)

    # 将数据重新塑形为布尔数组
    unpacked = bytes_array.reshape(shape)
    return unpacked


def image_to_str(img_np: np.ndarray, quality: float = 90.0) -> str:
    """
    将NumPy图像数组转换为base64编码的字符串。
    
    Args:
        img_np: 图像的NumPy数组
        quality: JPEG压缩质量，0-100，默认90
        
    Returns:
        base64编码的图像字符串
    """
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    retval, buffer = cv2.imencode(".jpg", img_np, encode_param)
    img_str = base64.b64encode(buffer).decode("utf-8")
    return img_str


def str_to_image(img_str: str) -> np.ndarray:
    """
    将base64编码的字符串转换回图像NumPy数组。
    
    Args:
        img_str: base64编码的图像字符串
        
    Returns:
        解码后的图像NumPy数组
    """
    img_bytes = base64.b64decode(img_str)
    img_arr = np.frombuffer(img_bytes, dtype=np.uint8)
    img_np = cv2.imdecode(img_arr, cv2.IMREAD_ANYCOLOR)
    return img_np


def send_request(url: str, **kwargs: Any) -> dict:
    """
    发送HTTP请求的包装函数，支持多次重试。
    
    Args:
        url: 请求的URL地址
        **kwargs: 请求参数
        
    Returns:
        服务器响应结果的字典
    """
    response = {}
    for attempt in range(10):  # 尝试最多10次
        try:
            response = _send_request(url, **kwargs)
            break  # 请求成功则跳出循环
        except Exception as e:
            if attempt == 9:  # 达到最大尝试次数
                print(e)
                exit()
            else:
                print(f"Error: {e}. Retrying in 20-30 seconds...")
                time.sleep(20 + random.random() * 10)  # 随机等待20-30秒后重试

    return response


def _send_request(url: str, **kwargs: Any) -> dict:
    """
    实际发送HTTP请求的内部函数，使用锁文件防止并发请求冲突。
    
    Args:
        url: 请求的URL地址
        **kwargs: 请求参数
        
    Returns:
        服务器响应结果的字典
    """
    # 创建锁文件目录
    lockfiles_dir = "lockfiles"
    if not os.path.exists(lockfiles_dir):
        os.makedirs(lockfiles_dir)
    
    # 生成锁文件名
    filename = url.replace("/", "_").replace(":", "_") + ".lock"
    filename = filename.replace("localhost", socket.gethostname())
    filename = os.path.join(lockfiles_dir, filename)
    
    try:
        while True:
            # 等待直到锁文件不存在
            while os.path.exists(filename):
                # 文件存在时，等待50ms再检查
                time.sleep(0.05)

                try:
                    # 如果锁文件超过120秒未修改，认为是过期锁，删除它
                    if time.time() - os.path.getmtime(filename) > 120:
                        os.remove(filename)
                except FileNotFoundError:
                    pass  # 文件可能已被其他进程删除

            # 创建锁文件并写入随机字符串
            rand_str = str(random.randint(0, 1000000))
            with open(filename, "w") as f:
                f.write(rand_str)
            time.sleep(0.05)
            
            # 验证锁文件是否正确写入
            try:
                with open(filename, "r") as f:
                    if f.read() == rand_str:
                        break  # 锁文件正确创建
            except FileNotFoundError:
                pass  # 重试锁文件创建

        # 准备请求负载，将NumPy数组转换为字符串
        payload = {}
        for k, v in kwargs.items():
            if isinstance(v, np.ndarray):
                payload[k] = image_to_str(v, quality=kwargs.get("quality", 90))
            else:
                payload[k] = v

        # 设置HTTP头
        headers = {"Content-Type": "application/json"}

        # 发送请求并等待响应
        start_time = time.time()
        while True:
            try:
                resp = requests.post(url, headers=headers, json=payload, timeout=1)
                if resp.status_code == 200:
                    result = resp.json()
                    break
                else:
                    raise Exception("Request failed")
            except (
                requests.exceptions.Timeout,
                requests.exceptions.RequestException,
            ) as e:
                print(e)
                if time.time() - start_time > 20:
                    raise Exception("Request timed out after 20 seconds")

        try:
            # 删除锁文件
            os.remove(filename)
        except FileNotFoundError:
            pass  # 文件可能已被删除

    except Exception as e:
        try:
            # 确保发生异常时也删除锁文件
            os.remove(filename)
        except FileNotFoundError:
            pass
        raise e  # 重新抛出异常

    return result
