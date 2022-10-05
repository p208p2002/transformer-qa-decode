import torch
from loguru import logger
from collections import namedtuple
import numpy as np

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0) # only difference

def to_list(tensor):
    return tensor.detach().cpu().tolist()


def check_segment_type_is_a(start_index, segment_embeddings):
    tag_segment_embeddings = segment_embeddings[start_index]
    if 0 in tag_segment_embeddings:
        return True
    return False


def check_has_skip_token(check_tokens, skip_tokens):
    for check_token in check_tokens:
        for skip_token in skip_tokens:
            if check_token == skip_token:
                return True
    return False


def get_best_indexes(logits, n_best_size):
    """Get the n-best logits from a list."""
    index_and_score = sorted(
        enumerate(logits), key=lambda x: x[1], reverse=True)

    best_indexes = []
    for i in range(len(index_and_score)):
        if i >= n_best_size:
            break
        best_indexes.append(index_and_score[i][0])
    return best_indexes


TagResult = namedtuple('TagResult', ['answer_span', 'token', 'local_token_start',
                       'local_token_end', 'global_context_start', 'global_context_end', 
                       'start_logit', 'end_logit','score'])

ClsLogit = namedtuple('ClsLogit', ['start_logit', 'end_logit'])


class TransformerQADecode():
    def __init__(self, model, tokenizer,is_squad_v2=False):
        self.model = model
        self.tokenizer = tokenizer
        self.is_squad_v2 = is_squad_v2

        if str(self.model.device) == 'cpu':
            logger.warning(
                "Model inference using cpu, consider use gpu for accelerate")

    @torch.no_grad()
    def __call__(
        self, 
        question: str, context: str, 
        max_length: int = 512,
        stride: int = 384, 
        n_best_size: int = 10, 
        min_answer_token_length: int = 1, 
        max_answer_token_length: int = 10,
        return_cls_logits = False
    ):
        self.model.eval()

        inputs = self.tokenizer(
            question,
            context,
            return_tensors="pt",
            max_length=max_length,
            truncation='only_second',
            stride=stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding=True
        )


        # if True:
        #     _input_ids = inputs['input_ids']
        #     for x in _input_ids:
        #         logger.debug(self.tokenizer.decode(x))
    

        has_token_type_ids = True if 'token_type_ids' in inputs.keys() else False
        answer_results = []
        cls_logits = []
        for i, (input_ids, attention_mask) in enumerate(zip(inputs.input_ids, inputs.attention_mask)):
            input_ids = input_ids.to(self.model.device)
            attention_mask = attention_mask.to(self.model.device)
            fregment_answer_results = []
            fregment_cls_logits = []
            if has_token_type_ids:
                model_output = self.model(
                    input_ids=input_ids.unsqueeze(0),
                    attention_mask=attention_mask.unsqueeze(0),
                    token_type_ids=inputs
                    .token_type_ids[i]
                    .unsqueeze(0)
                    .to(self.model.device)
                )
            else:
                model_output = self.model(
                    input_ids=input_ids.unsqueeze(0),
                    attention_mask=attention_mask.unsqueeze(0)
                )

            cls_start_logit = model_output.start_logits[0][0]
            cls_end_logit = model_output.end_logits[0][0]
            fregment_cls_logits.append(
                ClsLogit(
                    start_logit=cls_start_logit,
                    end_logit=cls_end_logit
                )
            )

            start_logits = to_list(model_output.start_logits)[0]
            end_logits = to_list(model_output.end_logits)[0]

            best_start_indexs = get_best_indexes(
                start_logits, n_best_size=n_best_size)
            best_end_indexs = get_best_indexes(
                end_logits, n_best_size=n_best_size)

            input_decode = self.tokenizer.convert_ids_to_tokens(input_ids)

            for start_index in best_start_indexs:
                for end_index in best_end_indexs:
                    answer_token = input_decode[start_index:end_index+1]
                    if(len(answer_token) < min_answer_token_length or len(answer_token) > max_answer_token_length):
                        continue
                    if self.is_squad_v2 and check_has_skip_token(check_tokens=answer_token, skip_tokens=[self.tokenizer.sep_token, self.tokenizer.pad_token]):
                        continue
                    if not self.is_squad_v2 and check_has_skip_token(check_tokens=answer_token, skip_tokens=[self.tokenizer.cls_token,self.tokenizer.sep_token, self.tokenizer.pad_token]):
                        continue
                    if has_token_type_ids and check_segment_type_is_a(start_index, inputs.token_type_ids[i]):
                        continue

                    offset_mapping = [
                        inputs.offset_mapping[i][start_index][0].item(),
                        inputs.offset_mapping[i][end_index][-1].item()
                    ]

                    fregment_answer_results.append(
                        TagResult(
                            token=answer_token,
                            answer_span=context[offset_mapping[0]:offset_mapping[1]],
                            local_token_start=start_index,
                            local_token_end=end_index,
                            global_context_start=offset_mapping[0],
                            global_context_end=offset_mapping[-1],
                            start_logit=start_logits[start_index],
                            end_logit=end_logits[end_index],
                            score=-1
                        )
                    )

            # compute score
            start_probs = softmax([x.start_logit for x in fregment_answer_results])
            end_probs = softmax([x.end_logit for x in fregment_answer_results])
            new_scores = [s*e for s,e in zip(start_probs,end_probs)]
            
            for i,new_score in enumerate(new_scores):
                fregment_answer_results[i] = TagResult(
                    *fregment_answer_results[i][:-1],
                    score = new_score
                )

            fregment_answer_results = sorted(fregment_answer_results,key=lambda x:x.score,reverse=True)
         
            # record
            answer_results.append(fregment_answer_results)
            cls_logits.append(fregment_cls_logits)
    
        if return_cls_logits:
            return answer_results,cls_logits
        return answer_results

