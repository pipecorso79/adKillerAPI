import base64
import json

def decode_data(json_data, key):
    base64_string = json_data[key]
    return base64.b64decode(base64_string)