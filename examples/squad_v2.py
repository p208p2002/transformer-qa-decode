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

