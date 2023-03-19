from typing import List
from torch.nn import Module
import warnings


def FederatedAverage(model_list: List[Module]) -> List[Module]:
    if len(model_list) is 1:
        warnings.warn("federated average does not work for one model")
        return model_list
    else:
        model_num = len(model_list)
        weight = model_list[0].state_dict()
        for key in weight:
            for model_id in range(1, model_num):
                weight[key] += model_list[model_id].state_dict()[key]
            weight[key] = (weight[key] / model_num).to(weight[key].dtype)
        for model_id in range(model_num):
            model_list[model_id].load_state_dict(weight)
        return model_list
