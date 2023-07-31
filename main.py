import jsonlines
import openai
if __name__ == '__main__':

    jsonl_savepath= 'beijing_gov_rp.jsonl'

    # 逐行读取jsonline文件
    with open(jsonl_savepath, "r", encoding="utf-8") as f:
        for item in jsonlines.Reader(f):
            print(item["article"])
            break


    # # 设置你的 API 密钥
    # openai.api_key = "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"
    #
    # # 调用 ChatGPT
    # response = openai.Completion.create(
    #     engine="davinci",  # 使用 "davinci" 引擎，但你也可以选择其他引擎
    #     prompt="Translate the following English text to French: 'Hello, how are you?'",
    #     max_tokens=50  # 限制响应的最大长度
    # )
    #
    # # 打印响应
    # print(response.choices[0].text.strip())

