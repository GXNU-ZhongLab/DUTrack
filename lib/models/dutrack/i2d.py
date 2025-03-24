import torch
from transformers import BertTokenizer
from transformers import BlipProcessor, BlipForConditionalGeneration

import torch.nn as nn


class descriptgenRefiner(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, blip_dir,bert_dir):
        super().__init__()
        self.processor = BlipProcessor.from_pretrained(blip_dir)
        self.model = BlipForConditionalGeneration.from_pretrained(blip_dir,torch_dtype=torch.float16).to("cuda")
        self.tokenizer = BertTokenizer.from_pretrained(bert_dir)

    def forward(self, image, cls):
        if cls is None:
            inputs = self.processor(image, return_tensors="pt").to("cuda", torch.float16)
        else:
            inputs = self.processor(image, cls, return_tensors="pt").to("cuda", torch.float16)
        out = self.model.generate(**inputs)
        descript = self.processor.decode(out[0], skip_special_tokens=True)
        return descript

