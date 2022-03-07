from transformer_qa_decode import TransformerQADecode
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import json

def flatten(t):
    return [item for sublist in t for item in sublist]

def named_tuple_to_dict(t):
    return [nt._asdict() for nt in t]

def get_top_k_by_start_logit(t:list,k):
    return sorted(t,key=lambda e:e['start_logit']*-1)[:k]

tokenizer = AutoTokenizer.from_pretrained("deepset/roberta-base-squad2")
model = AutoModelForQuestionAnswering.from_pretrained(
    "deepset/roberta-base-squad2")
    
# inference with gpu
# model.to('cuda')

qahl = TransformerQADecode(model=model, tokenizer=tokenizer)

question = "Which name is also used to describe the Amazon rainforest in English?"
context = """
The Amazon rainforest (Portuguese: Floresta Amazônica or Amazônia; Spanish: Selva Amazónica, Amazonía or usually Amazonia; French: Forêt amazonienne; Dutch: Amazoneregenwoud), also known in English as Amazonia or the Amazon Jungle, is a moist broadleaf forest that covers most of the Amazon basin of South America. This basin encompasses 7,000,000 square kilometres (2,700,000 sq mi), of which 5,500,000 square kilometres (2,100,000 sq mi) are covered by the rainforest. This region includes territory belonging to nine nations. The majority of the forest is contained within Brazil, with 60% of the rainforest, followed by Peru with 13%, Colombia with 10%, and with minor amounts in Venezuela, Ecuador, Bolivia, Guyana, Suriname and French Guiana. States or departments in four nations contain "Amazonas" in their names. The Amazon represents over half of the planet's remaining rainforests, and comprises the largest and most biodiverse tract of tropical rainforest in the world, with an estimated 390 billion individual trees divided into 16,000 species.
"""

result = flatten(qahl(question, context))
result = named_tuple_to_dict(result)
result = get_top_k_by_start_logit(result,10)

data = {
    "result":result,
    "context":context,
    "question":question
}

with open("dump_roberta.json","w",encoding="utf-8") as f:
    json.dump(data,f,ensure_ascii=False)



