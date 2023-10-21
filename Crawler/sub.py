import requests
from bs4 import BeautifulSoup
import jieba
# 要爬取的目标网站的URL
url = 'http://fjnews.fjsen.com/2022-10/09/content_31149826.htm'  # 将此处替换为你要爬取的网站URL

# 发送HTTP请求，获取网页内容
response = requests.get(url)
response.encoding = 'utf-8'
# 检查请求是否成功
if response.status_code == 200:
    # 使用Beautiful Soup解析网页内容
    soup = BeautifulSoup(response.text, 'html.parser')

    # 使用Beautiful Soup方法获取网页文本内容
    p_tag = soup.find_all('p')
    j_tags = soup.find_all('strong')
    for j_tag in j_tags:
        print(j_tag.text)
    # # 打印网页文本内容
    # for p in p_tag:
    #     word_list = list(jieba.cut(p.text))
    #     if len(word_list):
    #         if word_list[0] == '记者':
    #             print(p.text)
    #         elif word_list[0] == '童长峰' or '高榕':
    #             print(p.text)


else:
    print('请求失败，状态码：', response.status_code)
