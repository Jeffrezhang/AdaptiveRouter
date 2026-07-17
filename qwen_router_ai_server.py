from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch, uvicorn, asyncio

app = FastAPI()

BASE_MODEL_PATH = "finetune-selfcognition/qwen2.5-7b-base"
ADAPTER_PATH = "finetune-selfcognition/lora-adapter"

print("Loading QwenRouterAI...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_PATH, dtype=torch.float16, device_map="cpu")
model = PeftModel.from_pretrained(model, ADAPTER_PATH)
model.eval()
print("QwenRouterAI ready on port 8001!")

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: str
    messages: list[Message]
    max_tokens: int = 512

@app.post("/v1/chat/completions")
def chat(req: ChatRequest):
    from datetime import datetime
    messages = [{"role": m.role, "content": m.content} for m in req.messages]
    
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    
    inputs = tokenizer(text, return_tensors="pt")
    input_len = inputs.input_ids.shape[1]

    print(f"[{datetime.now().isoformat()}] [debug] REQUEST RECEIVED input_len={input_len}")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=req.max_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    output_len = outputs.shape[1] - input_len
    response = tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1]:],
        skip_special_tokens=True,
    )

    print(f"[{datetime.now().isoformat()}] [debug] GENERATION DONE input_len={input_len} output_len={output_len} response={response!r}")

    return {
        "id": "qwenrouterai-1",
        "object": "chat.completion",
        "model": req.model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": response},
            "finish_reason": "stop"
        }]
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)