import os

path = r'D:\netra-scene-description\netra_env\Lib\site-packages\flash_attn'
os.makedirs(path, exist_ok=True)

with open(os.path.join(path, '__init__.py'), 'w') as f:
    f.write('''def is_flash_attn_2_available():
    return False

def flash_attn_func(*args, **kwargs):
    raise NotImplementedError("Flash attention not available on CPU")

VERSION = "0.0.0"
''')

print('Created flash_attn stub package!')