import base64
import urllib
import requests

API_KEY = "ekVUZynbFWlWnTCq5xvhT08b"
SECRET_KEY = "sIe6W4rq7BWC7jt5M77UdFs9rkAn83NT"


def main():
    url = "https://aip.baidubce.com/rest/2.0/ocr/v1/accurate_basic?access_token=" + get_access_token()

    # image 可以通过 get_file_content_as_base64("C:\fakepath\20260224-2219-2-S V3.0(1).png",True) 方法获取
    payload = 'image='+get_file_content_as_base64(r"C:\Users\1\PyCharmMiscProject\black_region_output\right_table_crop.png",True)
    headers = {
        'Content-Type': 'application/x-www-form-urlencoded',
        'Accept': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload.encode("utf-8"))

    response.encoding = "utf-8"
    print(response.text)


def get_file_content_as_base64(path, urlencoded=False):
    """
    获取文件base64编码
    :param path: 文件路径
    :param urlencoded: 是否对结果进行urlencoded
    :return: base64编码信息
    """
    with open(path, "rb") as f:
        content = base64.b64encode(f.read()).decode("utf8")
        if urlencoded:
            content = urllib.parse.quote_plus(content)
    return content


def get_access_token():
    """
    使用 AK，SK 生成鉴权签名（Access Token）
    :return: access_token，或是None(如果错误)
    """
    url = "https://aip.baidubce.com/oauth/2.0/token"
    params = {"grant_type": "client_credentials", "client_id": API_KEY, "client_secret": SECRET_KEY}
    return str(requests.post(url, params=params).json().get("access_token"))


if __name__ == '__main__':
    main()