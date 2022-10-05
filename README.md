# Transformer QA Decode
A basic span-based decode strategy for transformer models, also provide local token mapping and global context mapping, which can present a [visualize demo](https://github.com/p208p2002/react-transformer-qa-decode-visualize).

## Usage
#### Install
```
pip install -U git+https://github.com/p208p2002/transformer-qa-decode.git
```
#### Example (SQuAD v1)
```python
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
#### Example (SQuAD v2)
```python
from transformer_qa_decode import TransformerQADecode
from transformers import AutoTokenizer, AutoModelForQuestionAnswering


tokenizer = AutoTokenizer.from_pretrained("deepset/roberta-base-squad2")
model = AutoModelForQuestionAnswering.from_pretrained(
    "deepset/roberta-base-squad2")

qahl = TransformerQADecode(model=model, tokenizer=tokenizer, is_squad_v2=True)

def answer_question(context,question):
    print("C:",context,"Q:",question)
    results,cls_logits = qahl(question, context,return_cls_logits=True)
    for stride_result,stride_cls_logits in zip(results,cls_logits):
        top_1 = stride_result[0]
        if top_1.global_context_start == 0 and top_1.global_context_end == 0:
            print("Unanswerable")
            print("-"*50)
            continue
        else:
            for result in stride_result[:3]:
                print("A:",result.answer_span)
            print("-"*50)


answer_question(
    question = "Where is the dog ?",
    context = "My name is Clara and I live in Berkeley."
)

answer_question(
    question = "What is my name?",
    context = "My name is Clara and I live in Berkeley."
)

"""
C: My name is Clara and I live in Berkeley. Q: Where is the dog ?
Unanswerable
--------------------------------------------------
C: My name is Clara and I live in Berkeley. Q: What is my name?
A: Clara
A: Clara and I live in Berkeley.
A: Clara and I live in Berkeley
--------------------------------------------------
"""
```

> See more use case in [examples](examples)
