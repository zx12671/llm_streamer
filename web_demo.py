from transformers import AutoModelForCausalLM, AutoTokenizer,TextStreamer
import streamlit as st
from streamlit_chat import message
import torch

st.set_page_config(
    page_title="ChatGLM-6b 演示",
    page_icon=":robot:"
)


# @st.cache_resource
# def get_model():
#     tokenizer = AutoTokenizer.from_pretrained("/mnt/lx/LMFlow/output_models/chatglm-6b", trust_remote_code=True)
#     model = AutoModel.from_pretrained("/mnt/lx/LMFlow/output_models/chatglm-6b", trust_remote_code=True).half().cuda()
#     model = model.eval()
#     return tokenizer, model

@st.cache_resource
def get_model():
    # checkpoint = "LaMini-GPT-1.5B"  #LaMini-GPT-124m
    checkpoint = "C:/Users/xiangli/workspace/LaMini-GPT-1.5B/LaMini-GPT-124m"  # LaMini-GPT-124m

    tokenizer = AutoTokenizer.from_pretrained(f"{checkpoint}")

    model = AutoModelForCausalLM.from_pretrained(f"{checkpoint}")
    model = model.eval()
    return tokenizer, model

tokenizer, model = get_model()
MAX_TURNS = 5
MAX_BOXES = MAX_TURNS * 2


@torch.no_grad()
def chat( model, tokenizer, query: str, history = None, max_length: int = 2048, num_beams=1,
         do_sample=True, top_p=0.7, temperature=0.95, logits_processor=None, **kwargs):
    if history is None:
        history = []
    # # if logits_processor is None:
    # #     logits_processor = LogitsProcessorList()
    # # logits_processor.append(InvalidScoreLogitsProcessor())
    # gen_kwargs = {"max_length": max_length, "num_beams": num_beams, "do_sample": do_sample, "top_p": top_p,
    #               "temperature": temperature, "logits_processor": logits_processor, **kwargs}
    if not history:
        prompt = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{query}\n\n### Response:"
    else:
        prompt = ""
        for i, (old_query, response) in enumerate(history):
            prompt += "[Round {}]\n问：{}\n答：{}\n".format(i, old_query, response)
        prompt += "[Round {}]\n问：{}\n答：".format(len(history), query)
    input_ids = tokenizer([prompt], return_tensors="pt", padding=True)
    outputs = model.generate(**input_ids, max_length=256, num_return_sequences=1)
    outputs = outputs.tolist()[0][len(input_ids["input_ids"][0]):]  # ref: modeling_chatglm.py chat
    response = tokenizer.decode(outputs, skip_special_tokens=False)
    history = history + [(query, response)]
    return response, history





def predict(input, max_length, top_p, temperature, history=None):
    # streamer = TextStreamer(tokenizer)
    # inputs = tokenizer([input], return_tensors="pt", padding=True)
    if history is None:
        history = []

    with container:
        if len(history) > 0:
            if len(history)>MAX_BOXES:
                history = history[-MAX_TURNS:]
            for i, (query, response) in enumerate(history):
                message(query, avatar_style="big-smile", key=str(i) + "_user")
                message(response, avatar_style="bottts", key=str(i))

        message(input, avatar_style="big-smile", key=str(len(history)) + "_user")
        st.write("AI正在回复:")
        with st.empty():
            # for response, history in model.stream_chat(tokenizer, input, history, max_length=max_length, top_p=top_p,
            #                                    temperature=temperature):
            #     query, response = history[-1]
                response, history = chat(model, tokenizer, input, history= history, max_length=max_length, top_p=top_p, temperature=temperature)
                query, response = history[-1]
                st.write(response)
    return history


container = st.container()

# create a prompt text for the text generation
prompt_text = st.text_area(label="用户命令输入",
            height = 100,
            placeholder="请在这儿输入您的命令")

max_length = st.sidebar.slider(
    'max_length', 0, 4096, 2048, step=1
)
top_p = st.sidebar.slider(
    'top_p', 0.0, 1.0, 0.6, step=0.01
)
temperature = st.sidebar.slider(
    'temperature', 0.0, 1.0, 0.95, step=0.01
)

if 'state' not in st.session_state:
    st.session_state['state'] = []

if st.button("发送", key="predict"):
    with st.spinner("AI正在思考，请稍等........"):
        # text generation
        st.session_state["state"] = predict(prompt_text, max_length, top_p, temperature, st.session_state["state"])
