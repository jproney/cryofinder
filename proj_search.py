import torch
from cryodrgn.lattice import Lattice
from cryodrgn.pose_search import rot_2d, interpolate
from cryodrgn import fft
import pickle

def translate_images(images, trans, lat=None, mask=None, input_hartley=True, output_hartley=True):
    """
    Translate images in Hartley/real space

    Args:
        images: Tensor of shape N x D x D, images in real or Hartley space
        trans: Tensor of translations to apply, T x 2 or N x T x 2
        lat: Optional Lattice object. If None, creates new Lattice with D+1 size
        mask: Optional mask for lattice coordinates. If None, uses circular mask
        input_hartley: Bool indicating if input images are in Hartley space
        output_hartley: Bool indicating if output should be in Hartley space

    Returns:
        Tensor of translated images with shape N x T x D x D, where T is number of translations.
        Output is in Hartley or real space based on output_hartley parameter.
    """

    if lat is None:
        lat = Lattice(images.shape[1]+1) #, extent=images.shape[1]//2)

    if not input_hartley:
        ht = fft.ht2_center(images)
        ht = fft.symmetrize_ht(ht)
    else:
        ht = images

    if len(trans.shape) == 2:
        trans = trans.unsqueeze(0)

    if mask is None:
        mask = lat.get_circular_mask((images.shape[1]) // 2)

    translated_flat = lat.translate_ht(ht.reshape(images.shape[0], -1)[:, mask], trans, mask)

    # outputs of translation in hartley space. Shape N x T x D x D
    trans_images = torch.zeros((images.shape[0], translated_flat.shape[1], ht.shape[1]**2), device=images.device)
    trans_images[..., mask] = translated_flat
    trans_images = trans_images.view((trans_images.shape[:-1] + ht.shape[1:]))

    if not output_hartley:
        trans_images = fft.iht2_center(trans_images[...,:-1,:-1])

    return trans_images

def rotate_images(images, rot, lat=None, mask=None, input_hartley=True, output_hartley=True, fast_rotate=False):
    """
    images - shape N x D x D, real space image
    rot - shape R, radians
    lat - optional Lattice object
    mask - optional mask for lattice
    input_hartley - bool, whether input is in hartley space
    output_hartley - bool, whether output should be in hartley space
    """
    if lat is None:
        lat = Lattice(images.shape[1]+1)

    if mask is None:
        mask = lat.get_circular_mask((images.shape[1]) // 2)

    if not input_hartley:
        ht = fft.ht2_center(images)
        ht = fft.symmetrize_ht(ht)
    else:
        ht = images

    if fast_rotate:
        # Rotate in hartley space
        rot_matrices = torch.stack([rot_2d(a, 2, ht.device) for a in rot], dim=0)
        lattice_coords = lat.coords[mask][:, :2].to(ht.device) 
        rot_coords = lattice_coords @ rot_matrices

        ht_flat = ht.view(ht.shape[0], -1)[:, mask]

        full_images_masked = torch.zeros_like(ht)
        full_images_masked.view(-1, ht.shape[-1]**2)[:, mask] = ht_flat

        rot_images = torch.zeros((ht.shape[0], len(rot), ht.shape[-1]*ht.shape[-1]), device=ht.device)

        for angle_idx, interp_coords in enumerate(rot_coords):
            interpolated = interpolate(full_images_masked, interp_coords)
            interpolated *= ht_flat.std(-1, keepdim=True) / interpolated.std(-1, keepdim=True)
            rot_images[:,angle_idx, mask] = interpolated
        rot_images = rot_images.view(ht.shape[0], len(rot), ht.shape[-1], ht.shape[-1])
    else:
        rot_images = lat.rotate(ht, rot) # M x R x D x D

    if not output_hartley:
        rot_images = fft.iht2_center(rot_images[...,:-1,:-1])

    return rot_images

def optimize_theta_trans(ref_images, query_images, trans, rot, fast_rotate=False, input_hartley=False, hartley_corr=True, lat=None, mask=None):
    """
    query_image - shape N x D x D, real space image
    ref_images - shape M x D x D, real space image
    trans - shape N x T x 2 or T x 2, cartesian
    rot - shape R, radians
    fast_rotate - do interpolation-based rotation, bool (needs to be done in hartley space)
    input_harley - is the input in hartley space?
    hartley_corr - do the correlation in hartley space
    """

    if trans is not None and len(trans.shape) == 2:
        trans = trans.unsqueeze(0)

    if lat is None:
        lat = Lattice(ref_images.shape[1]+1)
    
    if mask is None:
        mask = lat.get_circular_mask((ref_images.shape[1]) // 2)



    if trans is not None:
        # outputs of translation in hartley space. Shape N x T x D x D
        ref_trans_images = translate_images(ref_images, trans, lat, mask, input_hartley=input_hartley, output_hartley=hartley_corr)
    else:            
        if not input_hartley and hartley_corr:
            ref_ht = fft.ht2_center(ref_images)
            ref_ht = fft.symmetrize_ht(ref_ht)
            ref_trans_images = ref_ht
        else:
            ref_trans_images = ref_images

    if rot is not None:
        if not fast_rotate:
            rot = rot.to(query_images.device)
        query_rot_images = rotate_images(query_ht, rot, lat=lat, mask=mask, fast_rotate=fast_rotate, input_hartley=input_hartley, output_hartley=hartley_corr)
    else:
        if not input_hartley and hartley_corr:
            query_ht = fft.ht2_center(query_images)
            query_ht = fft.symmetrize_ht(query_ht)
            query_rot_images = query_ht
        else:
            query_rot_images = query_images

    query_expanded = query_rot_images.unsqueeze(0).unsqueeze(2)
    ref_expanded = ref_trans_images.unsqueeze(1).unsqueeze(3)
    print(query_expanded.shape)
    print(ref_expanded.shape)


    # Compute normalized cross correlation
    pairwise_corr = ((query_expanded * ref_expanded).sum(dim=(-1,-2)) / (
        torch.std(query_expanded, dim=(-1,-2)) * torch.std(ref_expanded, dim=(-1,-2)))).transpose(0,1)

    # Find best correlations in this chunk
    best_corr, best_indices = pairwise_corr.reshape(pairwise_corr.shape[0], -1).max(dim=-1)

    # Convert flattened indices to rotation, reference, translation indices
    best_indices = torch.stack(torch.unravel_index(best_indices,
                                                    (pairwise_corr.shape[1],  # references (chunk_size)
                                                    pairwise_corr.shape[2],  # translations
                                                    pairwise_corr.shape[3]   # rotations
                                                    )), dim=1)

    return best_corr, best_indices, pairwise_corr.amax(dim=(-1,-2))

def optimize_theta_trans_chunked(ref_images, query_images, trans, rot, chunk_size=100, fast_rotate=False, hartley_corr=True):
    """
    Memory-efficient version that processes reference images in chunks.
    
    Args:
        ref_images: shape M x D x D, real space images or list of paths pointing to image files
        query_images: shape N x D x D, real space images  
        trans: shape N x T x 2 or T x 2, cartesian coordinates
        rot: shape R, rotation angles in radians
        chunk_size: int, number of reference images to process at once, or number of chunk files to process at once
        fast_rotate: bool, whether to use fast rotation
        hartley_corr - do rotation in hartley space
        
    Returns:
        best_corr: shape N, best correlation for each query image
        best_indices: shape N x 3, indices of best (rotation, reference, translation) for each query
    """

    N = query_images.shape[0]
    device = query_images.device
    
    # Handle both tensor and list of paths
    if isinstance(ref_images, list):
        chunk_files = ref_images
    else:
        M = ref_images.shape[0]
        chunk_files = [None]
    
    corr_all = []

    # Initialize arrays to store best results
    best_corr = torch.full((N,), float('-inf'), device=device)
    best_indices = torch.zeros((N, 3), dtype=torch.long, device=device)
    
    # Pre-compute rotated query images
    lat = Lattice(query_images.shape[1]+1, device=device)
    mask = lat.get_circular_mask((query_images.shape[1]) // 2)

    if not fast_rotate:
        rot = rot.to(query_images.device)
    query_rot_images = rotate_images(query_images, rot, lat=lat, mask=mask, fast_rotate=fast_rotate, input_hartley=False, output_hartley=hartley_corr)
    
    # Process reference images in chunks
    global_offset = 0
    for cf in chunk_files:

        if cf is not None:
            print(cf)
            data = pickle.load(open(cf, 'rb'))
            ref_images = data['images'].to(device)
            M = ref_images.shape[0]

        for chunk_start in range(0, M, chunk_size):
            chunk_end = min(chunk_start + chunk_size, M)
            chunk_refs = ref_images[chunk_start:chunk_end].to(device)

            # put into hartley if needed
            if hartley_corr:
                chunk_refs = fft.ht2_center(chunk_refs)
                chunk_refs = fft.symmetrize_ht(chunk_refs)

            if trans is None:
                # just create a fake extra dim
                chunk_refs = chunk_refs.unsqueeze(1)

            # Get correlations for this chunk
            chunk_best_vals, chunk_best_indices, corr = optimize_theta_trans(chunk_refs, query_rot_images, trans, None, fast_rotate=fast_rotate, mask=mask, lat=lat, input_hartley=hartley_corr, hartley_corr=hartley_corr)        

            # Adjust reference indices to account for chunking
            chunk_best_indices[:,0] += chunk_start + global_offset
            
            # Update best results where this chunk had better correlations
            better_mask = chunk_best_vals > best_corr
            best_corr[better_mask] = chunk_best_vals[better_mask]
            best_indices[better_mask] = chunk_best_indices[better_mask]

            corr_all.append(corr)

        global_offset += M

    return best_corr, best_indices, torch.cat(corr_all, dim=-1)
