import torch


def get_max_length(input_ids, token_id):
    max_length = torch.sum(input_ids != token_id, dim=-1).max()
    return max_length
