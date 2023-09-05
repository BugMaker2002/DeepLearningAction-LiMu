import hashlib
import os
import tarfile
import zipfile
import requests
# a = 1
# assert a == 2, print("a不等于2")
# print("end")
# cache_dir = os.path.join('..', 'data')
# print(cache_dir)
def zwq(folder=None):
    data_dir = '/data/app'
    return os.path.join('.', folder) if folder else data_dir
print(zwq('new_app'))
