from transformer_qa_decode import TransformerQADecode
from transformers import AutoTokenizer, AutoModelForQuestionAnswering


tokenizer = AutoTokenizer.from_pretrained("abhilash1910/albert-squad-v2")
model = AutoModelForQuestionAnswering.from_pretrained(
    "abhilash1910/albert-squad-v2")

qahl = TransformerQADecode(model=model, tokenizer=tokenizer)
question = "What's my name?"
context = "My name is Clara and I live in Berkeley."
result = qahl(question, context)
print(result)
