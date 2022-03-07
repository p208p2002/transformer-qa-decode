# Transformer QA Decode
A basic span-based decode strategy for transformer models, also provide local token mapping and global context mapping, which can present a [visualize demo](#visualize).

## Usage
#### Install
```
pip install -U https://github.com/p208p2002/transformer-qa-decode.git
```
#### Example
```
from transformer_qa_decode import TransformerQADecode
from transformers import AutoTokenizer, AutoModelForQuestionAnswering


tokenizer = AutoTokenizer.from_pretrained("abhilash1910/albert-squad-v2")
model = AutoModelForQuestionAnswering.from_pretrained(
    "abhilash1910/albert-squad-v2")
    
# inference with gpu
# model.to('cuda')

qahl = TransformerQADecode(model=model, tokenizer=tokenizer)
question = "What's my name?"
context = "My name is Clara and I live in Berkeley."
result = qahl(question, context)
print(result)

"""
[[TagResult(answer_span='Clara', token=['▁clara'], local_token_start=11, local_token_end=11, global_context_start=11, global_context_end=16, start_logit=7.937160968780518, end_logit=7.778375625610352)...
"""
```
> See more use case in [examples](examples)

## Visualize
Here is a visualize component for react
 
[react-transformer-qa-decode-visualize](react-transformer-qa-decode-visualize)
