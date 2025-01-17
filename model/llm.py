import torch
import transformers

class LLM:
    """
    Base class for Language Model
    """
    def __init__(self, model_name):
        self.model_name = model_name

    def initialize_model(self):
        raise NotImplementedError("Subclasses should implement func initialize_model!")

    def generate_response(self, prompt):
        raise NotImplementedError("Subclasses should implement func generate_response!")
    

class Llama(LLM):
    """
    Llama class for generating responses
    """
    def __init__(self, model_name, model_id, max_new_tokens, device):
        super().__init__(model_name)
        self.device = device
        self.model = self.initialize_model(model_id, device)
        self.max_new_tokens = max_new_tokens

    def initialize_model(self, model_id, device):
        pipe = transformers.pipeline(
            "text-generation",
            model=model_id,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map=device,
        )
        return pipe

    def generate_response(self, message):

        terminators = [
            self.model.tokenizer.eos_token_id,
            self.model.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        completion = self.model(
            message,
            max_new_tokens=self.max_new_tokens,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
            pad_token_id = self.model.tokenizer.eos_token_id
        )

        return self._construct_assistant_message(completion)
    
    def _construct_assistant_message(self, completion):
        content = completion[0]["generated_text"][-1]["content"]
        return {"role": "assistant", "content": content}

    
class OpenAI(LLM):
    pass