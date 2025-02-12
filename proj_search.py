import torch
from cryodrgn.lattice import Lattice
from cryodrgn.pose_search import rot_2d, interpolate
from cryodrgn import fft


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
        lattice_coords = lat.coords[mask][:, :2] 
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

def optimize_theta_trans(ref_images, query_images, trans, rot, fast_rotate=False, fast_translate=False, refine_fast_translate=True, pre_rotated=False, max_trans=14, lat=None, mask=None):
    """
    query_image - shape N x D x D, real space image
    ref_images - shape M x D x D, real space image
    trans - shape N x T x 2 or T x 2, cartesian
    rot - shape R, radians
    fast_rotate - do interpolation-based rotation, bool
    """

    if len(trans.shape) == 2:
        trans = trans.unsqueeze(0)

    if lat is None:
        lat = Lattice(ref_images.shape[1]+1)
    
    if mask is None:
        mask = lat.get_circular_mask((ref_images.shape[1]) // 2)


    # make many translations of the references in hartley space
    ref_ht = fft.ht2_center(ref_images)
    ref_ht = fft.symmetrize_ht(ref_ht)

    if not fast_translate:
        # outputs of translation in hartley space. Shape N x T x D x D
        ref_trans_images = translate_images(ref_ht, trans, lat, mask)

    if not pre_rotated:
        # rotate the query in hartley space 
        query_ht = fft.ht2_center(query_images)
        query_ht = fft.symmetrize_ht(query_ht)

        # Use rotate_images function instead of duplicating rotation logic
        query_rot_images = rotate_images(query_ht, rot, lat=lat, mask=mask, fast_rotate=fast_rotate)
    else:
        query_rot_images = query_images

    if not fast_translate:
        query_expanded = query_rot_images.unsqueeze(0).unsqueeze(2)
        ref_expanded = ref_trans_images.unsqueeze(1).unsqueeze(3)

        # Compute normalized cross correlation in hartley space
        pairwise_corr = (query_expanded * ref_expanded).sum(dim=(-1,-2)) / (
            torch.std(query_expanded, dim=(-1,-2)) * torch.std(ref_expanded, dim=(-1,-2)))
    else:
        #frequency filter mask
        h, w = query_rot_images.shape[-2:]
        y, x = torch.meshgrid(torch.linspace(-1, 1, h), torch.linspace(-1, 1, w), indexing='ij')
        radius = torch.sqrt(x**2 + y**2)

        query_expanded = query_rot_images.unsqueeze(0)  * (radius < .15) # 1 x N x R x D x D
        ref_expanded = ref_ht.unsqueeze(1).unsqueeze(2)  * (radius < .15) # M x 1 x 1 x D x D

        # Compute cross power spectrum in Hartley space
        cross_power = query_expanded * ref_expanded / (torch.abs(query_expanded * ref_expanded) + 1e-8) # M x N x R x D x D

        # Convert to real space to find optimal alignment
        # Convert to real space and extract central region
        full_corr = fft.iht2_center(cross_power[...,:-1,:-1]) # M x N x R x D x D
        D = full_corr.shape[-1]
        start = (D - max_trans) // 2
        end = start + max_trans
        pairwise_corr = full_corr[..., start:end, start:end] # M x N x R x 30 x 30

        # after finding the pairs, refine translation with precise alignment
        if refine_fast_translate:
            maxcorr = pairwise_corr.amax(dim=(-1,-2, -3))
            bestref = ref_ht[maxcorr.argmax(dim=0)] # dim N

            ref_trans_images = translate_images(bestref, trans, lat, mask).unsqueeze(2) # N x T x 1 x D x D

            query_expanded = query_rot_images.unsqueeze(1) # N x 1 x R x D x D

            # Compute normalized cross correlation in hartley space
            pairwise_corr = (query_expanded * ref_trans_images).sum(dim=(-1,-2)) / (
                torch.std(query_expanded, dim=(-1,-2)) * torch.std(ref_trans_images, dim=(-1,-2))) # N x T x R


            # Find best correlations in this chunk
            best_corr, best_indices = pairwise_corr.reshape(pairwise_corr.shape[0], -1).max(dim=-1)


            # Convert flattened indices to rotation, reference, translation indices
            best_indices = torch.stack((maxcorr.argmax(dim=0),) + torch.unravel_index(best_indices,
                                                            (pairwise_corr.shape[1],  # translations
                                                            pairwise_corr.shape[2]   # rotations
                                                            )), dim=1)
            return maxcorr, bestref

        else:
            pairwise_corr = pairwise_corr.reshape(pairwise_corr.shape[:-2] + (-1,)).permute([0,1,3,2])


    # Find best correlations in this chunk
    best_corr, best_indices = pairwise_corr.transpose(0,1).reshape(pairwise_corr.shape[0], -1).max(dim=-1)

    # Convert flattened indices to rotation, reference, translation indices
    best_indices = torch.stack(torch.unravel_index(best_indices,
                                                    (pairwise_corr.shape[1],  # references (chunk_size)
                                                    pairwise_corr.shape[2],  # translations
                                                    pairwise_corr.shape[3]   # rotations
                                                    )), dim=1)

    return best_corr, best_indices

def optimize_theta_trans_chunked(ref_images, query_images, trans, rot, chunk_size=100, fast_rotate=False, fast_translate=False, refine_fast_translate=True, max_trans=14):
    """
    Memory-efficient version that processes reference images in chunks.
    
    Args:
        ref_images: shape M x D x D, real space images
        query_images: shape N x D x D, real space images  
        trans: shape N x T x 2 or T x 2, cartesian coordinates
        rot: shape R, rotation angles in radians
        chunk_size: int, number of reference images to process at once
        fast_rotate: bool, whether to use fast rotation
        
    Returns:
        best_corr: shape N, best correlation for each query image
        best_indices: shape N x 3, indices of best (rotation, reference, translation) for each query
    """
    M = ref_images.shape[0]
    N = query_images.shape[0]
    device = query_images.device
    
    # Initialize arrays to store best results
    best_corr = torch.full((N,), float('-inf'), device=device)
    best_indices = torch.zeros((N, 3), dtype=torch.long, device=device)
    
    # Pre-compute rotated query images
    lat = Lattice(query_images.shape[1]+1)
    mask = lat.get_circular_mask((query_images.shape[1]) // 2)
    query_ht = fft.ht2_center(query_images)
    query_ht = fft.symmetrize_ht(query_ht)
    query_rot_images = rotate_images(query_ht, rot, lat=lat, mask=mask, fast_rotate=fast_rotate)
    
    # Process reference images in chunks
    for chunk_start in range(0, M, chunk_size):
        chunk_end = min(chunk_start + chunk_size, M)
        chunk_refs = ref_images[chunk_start:chunk_end]
        
        # Get correlations for this chunk
        chunk_best_vals, chunk_best_indices = optimize_theta_trans(chunk_refs, query_rot_images, trans, rot, fast_rotate, fast_translate, refine_fast_translate=refine_fast_translate, max_trans=max_trans, mask=mask, lat=lat, pre_rotated=True)        

        # Adjust reference indices to account for chunking
        chunk_best_indices[:,0] += chunk_start
        
        # Update best results where this chunk had better correlations
        better_mask = chunk_best_vals > best_corr
        best_corr[better_mask] = chunk_best_vals[better_mask]
        best_indices[better_mask] = chunk_best_indices[better_mask]

    return best_corr, best_indices
