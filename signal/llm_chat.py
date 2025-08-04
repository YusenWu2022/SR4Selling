import os
from volcenginesdkarkruntime import Ark
from transformers import AutoTokenizer, AutoModelForCausalLM


def volc_chat(input):
    client = Ark(api_key="a2f4ef33-06da-4062-a808-f5a2287c54e5")
    completion = client.chat.completions.create(
        # 将 <Model> 替换为 Model ID（或者Endpoint ID）
        # model="deepseek-v3-241226", 
        model="deepseek-r1-distill-qwen-32b-250120",
        messages=[
            {"role": "system", "content": "你是擅长表格查询和编辑的工程师"},
            {"role": "user", "content": input}
        ]
    )
    return completion.choices[0].message.content

def extract_code_block(text):
    start_marker = "```python"
    end_marker = "```"
    start_index = text.find(start_marker)
    end_index = text.find(end_marker, start_index + len(start_marker))
    if start_index == -1 or end_index == -1:
        return text
    code_block = text[start_index + len(start_marker):end_index].strip()
    return code_block

def init_ds_model(device_id):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    from transformers import AutoTokenizer, AutoModelForCausalLM
    deepseek_14b_path = '/root/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-14B'
    tokenizer = AutoTokenizer.from_pretrained(deepseek_14b_path)
    model = AutoModelForCausalLM.from_pretrained(deepseek_14b_path, device_map="cuda:"+str(device_id))
    return model, tokenizer

def ds_14b_chat(input, model, tokenizer):

    messages = [
        {"role": "system", "content": "You are Qwen. You are a helpful assistant."},
        {"role": "user", "content": input},
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=4096,
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response

if __name__ == "__main__":
    model, tokenizer = init_ds_model()
    text = ds_14b_chat("你好，今天星期几", model, tokenizer)
    parts = text.split('</think>')
    result = parts[1].strip()
    print(result)