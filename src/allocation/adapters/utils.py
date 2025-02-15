import base64
def base64_to_bytes(base64_str):
    # 解碼 base64 字符串為 bytes
    return base64.b64decode(base64_str)
def bytes_to_base64(byte_data):
    # 編碼 bytes 為 base64 字符串
    return base64.b64encode(byte_data).decode('utf-8')