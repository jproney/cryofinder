# analyze data from the big run
import torch


def score_hits(topics, indices, terms):
    res = torch.zeros(indices.shape)
    # assumes first dim is different queries, last is hit ranking

    for i, (ts, ind) in enumerate(zip(terms, indices)):
        ind = ind.view((-1,))
        hit_topics = [topics[j][1].lower() for j in ind]
        correct = [any([all([a in ht for a in t.split(";")]) for t in ts]) for ht in hit_topics]
        res[i] = torch.tensor(correct, dtype=torch.float).view(res[i].shape)
    return res


def incorporate_postfiltered(results, mean_best_projection=True):

    nmaps = results['corr'].shape[0]
    nproj = results['corr'].shape[1]
    corr_all = results['corr'].clone().view([nmaps,nproj,-1,192])
    sorted_unique, sort_ids = torch.sort(results['unique_indices'], dim=-1)

    if mean_best_projection:
        retreival_indices = results['mbq_indices']
    
        positions = torch.searchsorted(sorted_unique, retreival_indices)
        mapped_ids = sort_ids[positions]
        corr_all[:,:, retreival_indices] = results['corr_pf'].view([nmaps,nproj,-1,192])[:,:, mapped_ids]
    else:
        for j in range(nproj):
            retreival_indices = results['bpqp_indices'][:,j]
    
            positions = torch.searchsorted(sorted_unique, retreival_indices)
            mapped_ids = sort_ids[positions]
            corr_all[:,j, retreival_indices] = results['corr_pf'].view([nmaps,nproj,-1,192])[:,j, mapped_ids]

    return corr_all

def select_best_maps(corr, strategy, maxk=64):
    if strategy == "worst_best":
        top_indices = corr.view.max(dim=-1)[0].min(dim=1)[0].topk(maxk, dim=-1)[1]
    elif strategy == "mean_best":
        top_indices = corr.max(dim=-1)[0].mean(dim=1).topk(maxk, dim=-1)[1]
    elif strategy == "best_best":
        top_indices = corr.max(dim=1)[0].max(dim=-1)[0].topk(maxk, dim=-1)[1]
    elif strategy == "per_proj":
        top_indices = corr.topk(maxk, dim=-1)[1]

    return top_indices
