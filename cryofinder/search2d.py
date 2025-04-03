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
    Rotate images in Hartley/real space. Can use either standard rotation via Lattice object
    or fast rotation using bilinear interpolation. Returns rotated images with shape N x R x D x D,
    where R is number of rotations.

    Args:
        images: Tensor of shape N x D x D, images in real or Hartley space
        rot: Tensor of rotation angles in radians, shape R
        lat: Optional Lattice object. If None, creates new Lattice with D+1 size
        mask: Optional mask for lattice coordinates. If None, uses circular mask
        input_hartley: Bool indicating if input images are in Hartley space
        output_hartley: Bool indicating if output should be in Hartley space
        fast_rotate: Bool indicating whether to use fast rotation via interpolation

    Returns:
        Tensor of rotated images with shape N x R x D x D, where R is number of rotations.
        Output is in Hartley or real space based on output_hartley parameter.
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

def optimize_theta_trans(ref_images, query_images, trans, rot, fast_rotate=False, input_hartley=False, hartley_corr=True, lat=None, mask=None, query_mask=None):
    """
    Optimizes over rotations and translations to find the best alignment between query and reference images.
    Supports both real space and Hartley space inputs/outputs, and provides options for fast rotation via interpolation.
    Returns correlation scores between query and reference images across all rotation and translation combinations.

    Args:
        ref_images: Reference images of shape M x D x D in real or Hartley space
        query_images: Query images of shape N x D x D in real or Hartley space  
        trans: Translation parameters of shape N x T x 2 or T x 2 in Cartesian coordinates
        rot: Rotation angles of shape R in radians
        fast_rotate: Whether to use fast rotation via interpolation (requires Hartley space)
        input_hartley: Whether input images are in Hartley space
        hartley_corr: Whether to compute correlations in Hartley space
        lat: Optional Lattice object for coordinate system
        mask: Optional mask for lattice coordinates
        query_mask: Optional mask of shape N x D x D for information-containing regions
    
    Returns:
        Returns a tuple of (corr, best_corr, best_indices):
            best_corr: Tensor of shape (N,) containing the maximum correlation score
                      for each query image across all references, translations and rotations
            
            best_indices: Tensor of shape (N x 3) containing the indices of the best
                         parameters that achieved the maximum correlation for each query.
                         best_indices[n] = [m,t,r] means reference m, translation t and 
                         rotation r gave the highest correlation for query n.
            pairwise_corr: Tensor of shape (N x M x T x R) containing correlation scores, where:
                N: Number of query images
                M: Number of reference images
                T: Number of translations (or 1 if trans=None)
                R: Number of rotations (or 1 if rot=None)
                Each element [n,m,t,r] represents the correlation between:
                    - Query image n rotated by rotation r
                    - Reference image m translated by translation t
                Higher correlation values indicate better alignment between the images.

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

    if query_mask is not None:
        query_mask = query_mask.unsqueeze(0).unsqueeze(2).unsqueeze(3)

    # Compute means
    if query_mask is not None:
        query_mean = (query_expanded * query_mask).sum(dim=(-1,-2), keepdim=True) / query_mask.sum(dim=(-1,-2), keepdim=True)
        ref_mean = (ref_expanded * query_mask).sum(dim=(-1,-2), keepdim=True) / query_mask.sum(dim=(-1,-2), keepdim=True)
    else:
        query_mean = query_expanded.mean(dim=(-1,-2), keepdim=True)
        ref_mean = ref_expanded.mean(dim=(-1,-2), keepdim=True)

    # Center the data
    query_centered = query_expanded - query_mean
    ref_centered = ref_expanded - ref_mean

    if query_mask is not None:
        query_centered *= query_mask
        ref_centered *= query_mask

    # Compute cross correlation
    pairwise_corr = ((query_centered * ref_centered).sum(dim=(-1,-2)) / (
        torch.sqrt((query_centered).pow(2).sum(dim=(-1,-2))) * torch.sqrt((ref_centered).pow(2).sum(dim=(-1,-2))))).transpose(0,1)

    # Find best correlations in this chunk
    best_corr, best_indices = pairwise_corr.reshape(pairwise_corr.shape[0], -1).max(dim=-1)

    # Convert flattened indices to rotation, reference, translation indices
    best_indices = torch.stack(torch.unravel_index(best_indices,
                                                    (pairwise_corr.shape[1],  # references (chunk_size)
                                                    pairwise_corr.shape[2],  # translations
                                                    pairwise_corr.shape[3]   # rotations
                                                    )), dim=1)

    return best_corr, best_indices, pairwise_corr

def optimize_theta_trans_chunked(ref_images, query_images, trans, rot, chunk_size=100, fast_rotate=False, hartley_corr=True, query_mask=None):
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
        query_mask: Optional[torch.Tensor], shape N x D x D, mask for query images (default: None)
    Returns:
        best_corr: shape N, best correlation for each query image
        best_indices: shape N x 3, indices of best (reference, translation, rotation) for each query
        corr: shape N x M x T x R, correlation values for each query image with each reference image at each translation and rotation
    """

    N = query_images.shape[0]
    device = query_images.device
    
    # Handle both tensor and list of paths
    if isinstance(ref_images, list):
        chunk_files = ref_images
    else:
        M = ref_images.shape[0]
        chunk_files = [None]
    

    # Initialize arrays to store best results
    best_corr = torch.full((N,), float('-inf'))
    best_indices = torch.zeros((N, 3), dtype=torch.long)
    corr_all = []

    # Pre-compute rotated query images
    lat = Lattice(query_images.shape[1]+1, device=device)
    mask = lat.get_circular_mask((query_images.shape[1]) // 2)

    if not fast_rotate:
        rot = rot.to(query_images.device)
    query_rot_images = rotate_images(query_images, rot, lat=lat, mask=mask, fast_rotate=fast_rotate, input_hartley=False, output_hartley=hartley_corr)
    

    # Pre-allocate a chunk of memory
    chunk_refs = torch.empty(chunk_size, *ref_images.shape[1:], device=device)

    # Process reference images in chunks
    global_offset = 0
    for cf in chunk_files:

        if cf is not None:
            print(f"Loading {cf}")
            data = pickle.load(open(cf, 'rb'))
            ref_images = data['images'].to(device)
            M = ref_images.shape[0]

        for chunk_start in range(0, M, chunk_size):
            chunk_end = min(chunk_start + chunk_size, M)
            chunk_refs[:chunk_end-chunk_start].copy_(ref_images[chunk_start:chunk_end])

            # put into hartley if needed
            if hartley_corr:
                ref_input = fft.ht2_center(chunk_refs[:chunk_end-chunk_start])
                ref_input = fft.symmetrize_ht(ref_input)
            else:
                ref_input = chunk_refs[:chunk_end-chunk_start]

            if trans is None:
                # just create a fake extra dim
                ref_input = ref_input.unsqueeze(1)

            # Get correlations for this chunk
            chunk_best_vals, chunk_best_indices, corr = optimize_theta_trans(ref_input, query_rot_images, trans, None, fast_rotate=fast_rotate, mask=mask, lat=lat, input_hartley=hartley_corr, hartley_corr=hartley_corr, query_mask=query_mask)        
            chunk_best_vals = chunk_best_vals.cpu()
            chunk_best_indices = chunk_best_indices.cpu()
            corr = corr.cpu()

            # Adjust reference indices to account for chunking
            chunk_best_indices[:,0] += chunk_start + global_offset
            
            # Update best results where this chunk had better correlations
            better_mask = chunk_best_vals > best_corr
            best_corr[better_mask] = chunk_best_vals[better_mask]
            best_indices[better_mask] = chunk_best_indices[better_mask]

            corr_all.append(corr)


        global_offset += M

    return best_corr, best_indices, torch.cat(corr_all, dim=1)
