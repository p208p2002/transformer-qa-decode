from transformer_qa_decode import TransformerQADecode
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

def flatten(t):
    return [item for sublist in t for item in sublist]

def named_tuple_to_dict(t):
    return [nt._asdict() for nt in t]

def get_top_k_by_start_logit(t:list,k):
    return sorted(t,key=lambda e:e['start_logit']*-1)[:k]


tokenizer = AutoTokenizer.from_pretrained("abhilash1910/albert-squad-v2")
model = AutoModelForQuestionAnswering.from_pretrained(
    "abhilash1910/albert-squad-v2")
    
# inference with gpu
# model.to('cuda')

qahl = TransformerQADecode(model=model, tokenizer=tokenizer)
question = "What's my name?"
context = "My name is Clara and I live in Berkeley."

result = flatten(qahl(question, context))
result = named_tuple_to_dict(result)
result = get_top_k_by_start_logit(result,10)

for r in result:
    print(r)



