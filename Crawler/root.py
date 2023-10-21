import requests
from bs4 import BeautifulSoup

# 要爬取的目标网站的URL
url = 'http://www.fjsen.com/a/gov/node_16712.htm'  # 将此处替换为你要爬取的网站URL

# 发送HTTP请求，获取网页内容
response = requests.get(url)

# 检查请求是否成功
if response.status_code == 200:
    # 使用Beautiful Soup解析网页内容
    soup = BeautifulSoup(response.text, 'html.parser')

    # 在这里可以使用Beautiful Soup方法来提取需要的数据
    keyword = 'content'  # 将此处替换为你要查找的关键词

    links_with_keyword = soup.find_all('a', href=lambda href: href and keyword in href)

    # 打印所有链接
    for link in links_with_keyword:
        print(link.get('href'))
else:
    print('请求失败，状态码：', response.status_code)
