# 标准库
import hashlib as _hashlib
import os
# 第三方库
from Crypto.Cipher import AES  # pycryptodome


class Hash:
    """Hashing String And File"""
    @staticmethod
    def __gene_hash_obj(hash_type):
        if hash_type == 1:
            return _hashlib.sha1()
        elif hash_type == 224:
            return _hashlib.sha224()
        elif hash_type == 256:
            return _hashlib.sha256()
        elif hash_type == 384:
            return _hashlib.sha384()
        elif hash_type == 512:
            return _hashlib.sha512()
        elif hash_type == 5:
            return _hashlib.md5()
        elif hash_type == 3.224:
            return _hashlib.sha3_224()
        elif hash_type == 3.256:
            return _hashlib.sha3_256()
        elif hash_type == 3.384:
            return _hashlib.sha3_384()
        elif hash_type == 3.512:
            return _hashlib.sha3_512()
        else:
            raise Exception('类型错误, 初始化失败')

    @staticmethod
    def file_hash(path, hash_type):
        """计算文件哈希
        :param path: 文件路径
        :param hash_type: 哈希算法类型
            1       sha-1
            224     sha-224
            256      sha-256
            384     sha-384
            512     sha-512
            5       md5
            3.256   sha3-256
            3.384   sha3-384
            3.512   sha3-512
        """
        hashObj = Hash.__gene_hash_obj(hash_type)
        if os.path.isfile(path):
            try:
                with open(path, "rb") as f:
                    for byte_block in iter(lambda: f.read(1048576), b""):
                        hashObj.update(byte_block)
                    return hashObj.hexdigest()
            except Exception as e:
                raise Exception('%s计算哈希出错: %s' % (path, e))
        else:
            raise Exception('路径错误, 没有指向文件: "%s"')

    @staticmethod
    def str_hash(str_: str, hash_type, charset='utf-8'):
        """计算字符串哈希
        :param str_: 字符串
        :param hash_type: 哈希算法类型
        :param charset: 字符编码类型
            1       sha-1
            224     sha-224
            256      sha-256
            384     sha-384
            512     sha-512
            5       md5
            3.256   sha3-256
            3.384   sha3-384
            3.512   sha3-512
        """
        hashObj = Hash.__gene_hash_obj(hash_type)
        bstr = str_.encode(charset)
        hashObj.update(bstr)
        return hashObj.hexdigest()

    @staticmethod
    def bytes_hash(bytes_: bytes, hash_type):
        """计算字节串哈希
        :param bytes_: 字节串
        :param hash_type: 哈希算法类型
            1       sha-1
            224     sha-224
            256      sha-256
            384     sha-384
            512     sha-512
            5       md5
            3.256   sha3-256
            3.384   sha3-384
            3.512   sha3-512
        """
        hashObj = Hash.__gene_hash_obj(hash_type)
        hashObj.update(bytes_)
        return hashObj.hexdigest()


class SimpleAES_StringCrypto:
    """
    在线加密解密见https://www.ssleye.com/aes_cipher.html
    key: sha256(secret_key)[0:32]
    iv: sha256(secret_key)[32:48]
    mode: CBC
    padding: pkcs7padding
    charset: utf-8
    encode: Hex
    """

    def __init__(self, secret_key: str):
        """
        :param secret_key: 密钥
        """
        self.charset = 'utf-8'

        hash = _hashlib.sha256()
        hash.update(secret_key.encode(self.charset))
        keyhash = hash.hexdigest()

        self.key = keyhash[0:32]
        self.iv = keyhash[32:48]
        print("AesCrypto initialization successful!\nkey: %s\niv: %s\nmode: CBC\npadding: pkcs7padding\ncharset: %s\nencode: Hex\n----------" %
              (self.key, self.iv, self.charset))
        self.key = self.key.encode(self.charset)
        self.iv = self.iv.encode(self.charset)

    def encrypt(self, text):
        """加密"""
        cipher = AES.new(self.key, AES.MODE_CBC, self.iv)

        text = self.pkcs7padding(text)  # 填充
        text = text.encode(self.charset)  # 编码
        text = cipher.encrypt(text)  # 加密
        text = text.hex()  # Hex编码
        return text

    def decrypt(self, text):
        """解密"""
        cipher = AES.new(self.key, AES.MODE_CBC, self.iv)

        text = bytes.fromhex(text)  # Hex解码
        text = cipher.decrypt(text)  # 解密
        text = text.decode(self.charset)  # 解码
        text = self.pkcs7unpadding(text)  # 删除填充
        return text

    def pkcs7padding(self, text: str):
        """明文使用PKCS7填充"""
        remainder = 16 - len(text.encode(self.charset)) % 16
        return str(text + chr(remainder) * remainder)

    def pkcs7unpadding(self, text: str):
        """去掉填充字符"""
        return text[:-ord(text[-1])]
