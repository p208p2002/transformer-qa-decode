from transformer_qa_decode import TransformerQADecode
from transformers import AutoTokenizer, AutoModelForQuestionAnswering


tokenizer = AutoTokenizer.from_pretrained("deepset/roberta-base-squad2")
model = AutoModelForQuestionAnswering.from_pretrained(
    "deepset/roberta-base-squad2")

# inference with gpu
# model.to('cuda')

qahl = TransformerQADecode(model=model, tokenizer=tokenizer)
question = "What's my name?"
context = "My name is Clara and I live in Berkeley."
result = qahl(question, context)
print(result)
