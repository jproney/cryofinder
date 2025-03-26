# analyze data from the big run
import torch


def score_hits(topics, indices, terms):
    res = torch.zeros(indices.shape)
    # assumes first dim is different queries, last is hit ranking

    for i, (ts, ind) in enumerate(zip(terms, indices)):
        ind = ind.view((-1,))
        hit_topics = [topics[j.item()][1].lower() for j in ind]
        correct = [any([all([a in ht for a in t.split(";")]) for t in ts]) for ht in hit_topics]
        res[i] = torch.tensor(correct, dtype=torch.float).view(res[i].shape)
    return res


def incorporate_postfiltered(results, mean_best_projection=True):

    nmaps = results['corr'].shape[0]
    nproj = results['corr'].shape[1]
    corr_all = results['corr'].clone().view([nmaps,nproj,-1,192])
    retreival_indices = results['mbq_indices'] if mean_best_projection else results['bpqp_indices']

    for i in range(nmaps):
        sorted_unique, sort_ids = torch.sort(results['unique_indices'][i], dim=-1)    
        positions = torch.searchsorted(sorted_unique, retreival_indices[i])
        mapped_ids = sort_ids[positions]

        if mean_best_projection:
            corr_all[i,:, retreival_indices[i]] = results['corr_pf'][i].view([nproj,-1,192])[:, mapped_ids]
        else:
            corr_all[i, torch.arange(nproj).unsqueeze(1), retreival_indices[i]] = results['corr_pf'][i].view([nproj,-1,192])[torch.arange(nproj).unsqueeze(1), mapped_ids]

    return corr_all

def select_best_maps(corr, strategy, maxk=64):
    if strategy == "worst_best":
        top_indices = corr.max(dim=-1)[0].min(dim=1)[0].topk(maxk, dim=-1)[1]
    elif strategy == "mean_best":
        top_indices = corr.max(dim=-1)[0].mean(dim=1).topk(maxk, dim=-1)[1]
    elif strategy == "best_best":
        top_indices = corr.max(dim=1)[0].max(dim=-1)[0].topk(maxk, dim=-1)[1]
    elif strategy == "per_proj":
        top_indices = corr.max(dim=-1)[0].topk(maxk, dim=-1)[1]

    return top_indices
