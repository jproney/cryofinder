import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from cryodrgn import ctf
from cryodrgn.lattice import Lattice
from cryofinder.search2d import rotate_images, translate_images

# corrupt a batch of particle images with simulated noise
def corrupt_with_ctf(batch_ptcls, batch_ctf_params, snr1, snr2, freqs, b_factor=None):

    stds = batch_ptcls.std(dim=(-1,-2))

    batch_ptcls += torch.randn(batch_ptcls.shape) * (stds / torch.sqrt(snr1)).view([-1,1,1])


    batch_ptcls = torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(batch_ptcls, dim=(-1, -2))), dim=(-1, -2))
    batch_ptcls = batch_ptcls.real - batch_ptcls.imag  # centered hartley transform
    ctf_weights = ctf.compute_ctf(freqs, *torch.split(batch_ctf_params, 1, dim=1), bfactor=b_factor).view(-1, batch_ptcls.shape[-1], batch_ptcls.shape[-1])
    batch_ptcls *= ctf_weights
    batch_ptcls = torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(batch_ptcls, dim=(-1, -2))), dim=(-1, -2))
    batch_ptcls = batch_ptcls.real - batch_ptcls.imag  

    batch_ptcls += torch.randn(batch_ptcls.shape)  * (stds / torch.sqrt(snr2)).view([-1,1,1])
    return batch_ptcls


class ContrastiveProjectionDataset(Dataset):
    def __init__(self, images, phis, thetas, object_ids, pos_angle_threshold=30, pclean=0.3, snr1=[1.5], 
                 dfu=[10000], Apix=5.0, ang=0.0, kv=300, cs=2.7, wgh=0.1, ps=0.0, proj_per_obj=192, img_size=128, p_hard=0.0,
                 obj_distance_matrix=None, pos_sim_threshold=0.4, neg_sim_threshold=0.35):
        """
        Dataset for contrastive learning of image projections.
        
        Args:
            images: Tensor of shape (num_objects, num_projections, img_size, img_size)
            phis: Tensor of shape (num_objects, num_projections) containing phi angles in degrees
            thetas: Tensor of shape (num_objects, num_projections) containing theta angles in degrees  
            object_ids: Tensor of shape (num_objects,) containing integer IDs for each 3D object
            pos_angle_threshold: Maximum angular difference in degrees for positive pairs
            snr1: List of signal-to-noise ratios for structural noise. snr2 is snr1 / 3
            dfu: List of defocus values in Angstroms for u-axis. dfv is dfu + 500
            Apix: Pixel size in Angstroms
            ang: Astigmatism angle in degrees
            kv: Microscope voltage in kV
            cs: Spherical aberration in mm
            wgh: Amplitude contrast ratio
            ps: Phase shift in radians
            obj_distance_matrix: Matrix of shape (num_objects, num_objects) containing similarity scores between objects
            pos_sim_threshold: Minimum similarity threshold for positive pairs (default 0.5)
            neg_sim_threshold: Maximum similarity threshold for negative pairs (default 0.4)
        """
        self.images = images
        self.phis = phis 
        self.thetas = thetas
        self.object_ids = object_ids
        self.pos_threshold = pos_angle_threshold
        self.snr1 = snr1
        self.dfu = dfu
        self.Apix = Apix
        self.ang = ang
        self.kv = kv
        self.cs = cs
        self.wgh = wgh
        self.ps = ps
        self.pclean = pclean
        self.proj_per_obj = proj_per_obj
        self.img_size = img_size
        self.p_hard = p_hard
        self.obj_distance_matrix = obj_distance_matrix
        self.pos_sim_threshold = pos_sim_threshold
        self.neg_sim_threshold = neg_sim_threshold

        # Create a Lattice object for transformations
        self.lat = Lattice(img_size + 1)
        self.mask = self.lat.get_circular_mask(img_size // 2)

        # prepare frequency lattice
        freqs = torch.arange(-img_size//2, img_size//2) / (self.Apix * img_size)
        x0, x1 = torch.meshgrid(freqs, freqs)
        self.freqs = torch.stack([x0.ravel(), x1.ravel()], dim=1)

    def __len__(self):
        return len(self.images) * self.proj_per_obj

    def __getitem__(self, idx):
        """
        Returns:
            anchor_img: The anchor image
            pos_img: A positive pair image (same object or similar object, similar angle)
            neg_img: A negative pair image (different object with low similarity)
        """
        # Determine object and projection indices
        obj_idx = idx // self.proj_per_obj
        proj_id = idx % self.proj_per_obj

        # Get anchor image and its angles
        anchor_img = self.images[obj_idx, proj_id]
        anchor_obj = self.object_ids[obj_idx]
        theta1, phi1 = self.thetas[obj_idx, proj_id], self.phis[obj_idx, proj_id]

        # Determine if we should use the distance matrix for sampling
        if self.obj_distance_matrix is not None:
            # Get similarity scores for the anchor object
            similarities = self.obj_distance_matrix[obj_idx]
            
            # Find objects with similarity above threshold for positive pairs
            # Exclude the anchor object itself to prioritize different objects
            pos_obj_candidates = torch.where((similarities >= self.pos_sim_threshold) & 
                                            (torch.arange(len(similarities)) != obj_idx))[0]
            
            # If no different objects found with high similarity, fall back to same object
            if len(pos_obj_candidates) == 0:
                pos_obj_idx = obj_idx
            else:
                # Sample from candidate objects (different from anchor)
                pos_obj_idx = pos_obj_candidates[torch.randint(len(pos_obj_candidates), (1,))].item()
            
            # Find objects with similarity below threshold for negative pairs
            neg_obj_candidates = torch.where(similarities <= self.neg_sim_threshold)[0]
            
            # If no candidates found, use random object different from anchor
            if len(neg_obj_candidates) == 0:
                neg_obj_idx = torch.randint(len(self.object_ids), (1,)).item()
                while neg_obj_idx == obj_idx:  # Ensure different object
                    neg_obj_idx = torch.randint(len(self.object_ids), (1,)).item()
            else:
                # Sample from candidate objects
                neg_obj_idx = neg_obj_candidates[torch.randint(len(neg_obj_candidates), (1,))].item()
        else:
            # If no distance matrix provided, use same object for positive and different for negative
            pos_obj_idx = obj_idx
            
            # Sample random object for negative
            neg_obj_idx = torch.randint(len(self.object_ids), (1,)).item()
            while neg_obj_idx == obj_idx:  # Ensure different object
                neg_obj_idx = torch.randint(len(self.object_ids), (1,)).item()

        # For positive pair, if using same object, select based on angular distance
        if pos_obj_idx == obj_idx:
            # Calculate angular distances for positive pair selection
            angle_dists = torch.arccos(
                torch.sin(theta1) * torch.sin(self.thetas[obj_idx]) * torch.cos(phi1 - self.phis[obj_idx]) + 
                torch.cos(theta1) * torch.cos(self.thetas[obj_idx])
            )
            
            # Create mask of valid positive pairs (within threshold and not same projection)
            valid_mask = (angle_dists <= self.pos_threshold * torch.pi/180)
            valid_mask[proj_id] = False  # Exclude the anchor projection
            
            # Sample from valid indices
            valid_indices = torch.where(valid_mask)[0]
            if len(valid_indices) > 0:
                pos_proj_idx = valid_indices[torch.randint(len(valid_indices), (1,))].item()
            else:
                # If no valid indices, choose a random projection from the same object
                pos_proj_idx = torch.randint(self.images.shape[1], (1,)).item()
                while pos_proj_idx == proj_id:  # Ensure different projection
                    pos_proj_idx = torch.randint(self.images.shape[1], (1,)).item()
        else:
            # If using different but similar object, choose random projection
            pos_proj_idx = torch.randint(self.images.shape[1], (1,)).item()
        
        # Get positive image
        pos_img = self.images[pos_obj_idx, pos_proj_idx]
        pos_obj = self.object_ids[pos_obj_idx]
        
        # With probability p_hard, select hard negative from same object
        if torch.rand(1).item() < self.p_hard and self.obj_distance_matrix is None:
            # Create mask of valid hard negatives (beyond 2x angular threshold)
            angle_dists = torch.arccos(
                torch.sin(theta1) * torch.sin(self.thetas[obj_idx]) * torch.cos(phi1 - self.phis[obj_idx]) + 
                torch.cos(theta1) * torch.cos(self.thetas[obj_idx])
            )
            valid_mask = (angle_dists > self.pos_threshold * torch.pi/180 * 2)
            
            # Calculate cross correlations between anchor and all projections of same object
            anchor_flat = anchor_img.flatten()
            all_projs = self.images[obj_idx].reshape(self.proj_per_obj, -1)
            cross_corrs = F.cosine_similarity(anchor_flat, all_projs)
            
            # Get indices of bottom 25% correlations that also satisfy angular threshold
            _, indices = torch.sort(cross_corrs)
            hard_neg_candidates = indices[valid_mask[indices]][:self.proj_per_obj//4]
            
            # Only proceed with hard negative if we found valid candidates
            if len(hard_neg_candidates) > 0:
                neg_proj_idx = hard_neg_candidates[torch.randint(len(hard_neg_candidates), (1,))].item()
                neg_img = self.images[obj_idx, neg_proj_idx]
                neg_obj = self.object_ids[obj_idx]
            else:
                # Fall back to different object if no valid hard negatives found
                neg_proj_idx = torch.randint(self.images.shape[1], (1,)).item()
                neg_img = self.images[neg_obj_idx, neg_proj_idx]
                neg_obj = self.object_ids[neg_obj_idx]
        else:
            # Get negative image from selected object
            neg_proj_idx = torch.randint(self.images.shape[1], (1,)).item()
            neg_img = self.images[neg_obj_idx, neg_proj_idx]
            neg_obj = self.object_ids[neg_obj_idx]

        # Sample CTF parameters for anchor, positive and negative images
        anchor_ctf = torch.zeros(9)
        pos_ctf = torch.zeros(9)
        neg_ctf = torch.zeros(9)

        # Set D and pixel size (first two params)
        anchor_ctf[0] = self.img_size
        anchor_ctf[1] = self.Apix
        pos_ctf[0] = self.img_size
        pos_ctf[1] = self.Apix
        neg_ctf[0] = self.img_size
        neg_ctf[1] = self.Apix

        # Randomly sample remaining CTF params for each image
        for i, param_list in enumerate([self.snr1, 0.0, self.dfu, 0.0, self.ang, self.kv,
                                      self.cs, self.wgh, self.ps]):
            if isinstance(param_list, list):
                anchor_ctf[i] = param_list[torch.randint(len(param_list), (1,))]
                pos_ctf[i] = param_list[torch.randint(len(param_list), (1,))]
                neg_ctf[i] = param_list[torch.randint(len(param_list), (1,))]
            else:
                anchor_ctf[i] = param_list
                pos_ctf[i] = param_list
                neg_ctf[i] = param_list

        for ctf in (anchor_ctf, pos_ctf, neg_ctf):
            ctf[1] = ctf[0] / 3
            ctf[3] = ctf[2] + 500

        # Stack images and CTF params
        images = torch.stack([anchor_img, pos_img, neg_img], dim=0)
        ctf_params = torch.stack([anchor_ctf, pos_ctf, neg_ctf], dim=0)

        return images, ctf_params, torch.tensor([anchor_obj, pos_obj, neg_obj])

    @staticmethod
    def collate_fn(batch, lat, mask, freqs, normalize=True, ctf_corrupt=False, noise=True):
        """
        Custom collate function to corrupt batches of triplet images with CTF and noise
        Args:
            batch: List of (images, ctf_params, pos_dist) tuples
        Returns:
            Corrupted (images, ctf_params, pos_dist) tuples
        """
        # Stack all images and CTF params from batch
        images = torch.stack([x[0] for x in batch])  # Shape: (batch, 3, D, D)
        ctf_params = torch.stack([x[1] for x in batch])  # Shape: (batch, 3, 9) 
        obj_ids = torch.stack([x[2] for x in batch])  # Shape: (batch,)

        B, N, D, _ = images.shape
        
        # Reshape to (batch*3, D, D) for corruption
        images = images.view(-1, D, D)
        ctf_params = ctf_params.view(-1, 9)

        # Randomly generate rotations and translations
        rotations = torch.rand(B * N) * 2 * torch.pi  # Random rotations in radians
        translations = torch.randint(-7, 8, (B * N, 2)).to(torch.float)  # Random translations within a range

        # Apply random rotations and translations
        images = rotate_images(images, rotations, lat=lat, mask=mask, input_hartley=False, output_hartley=False).diagonal().permute([2,0,1])
        
        images = translate_images(images, translations, lat=lat, mask=mask, input_hartley=False, output_hartley=False).diagonal().permute([2,0,1])
        

        if normalize:
            # normalize every image individually and constrain the range of values
            images = torch.clamp((images - images.mean(dim=(-1,-2), keepdim=True)) / images.std(dim=(-1,-2), keepdim=True), -5, 15)


        # Corrupt part of batch
        if ctf_corrupt:
            corrupted = corrupt_with_ctf(images, ctf_params[:,2:], ctf_params[:,0], ctf_params[:,1], freqs)
        elif noise:
            # Reshape to apply transformations independently per image
            corrupted = images.clone()  # Shape: (B, 3, 128, 128)
            
            # Add Gaussian noise with probability p_noise
            p_noise = 0.75
            noise_mask = (torch.rand(corrupted.shape[0]) < p_noise).to(corrupted.device)
            # Generate random noise levels between 0.0 and 0.25 for each image
            noise_levels = (torch.rand(corrupted.shape[0]) * 0.25).view(-1, 1, 1).to(corrupted.device)
            noise = torch.randn_like(corrupted) * noise_levels  # Random noise with varying levels
            corrupted[noise_mask] = corrupted[noise_mask] + noise[noise_mask]

            # Apply random contrast adjustment with probability p_contrast 
            p_contrast = 0.75
            contrast_mask = (torch.rand(corrupted.shape[0]) < p_contrast).to(corrupted.device)
            scales = (torch.rand(corrupted.shape[0]).to(corrupted.device) + 0.5).view(-1, 1, 1)  # Random scale in [0.5, 1.5]
            shifts = ((torch.rand(corrupted.shape[0]).to(corrupted.device) - 0.5) * 0.5).view(-1, 1, 1)  # Random shift in [-0.5, 0.5]
            corrupted[contrast_mask] = corrupted[contrast_mask] * scales[contrast_mask] + shifts[contrast_mask]
            
            # Reshape back to original dimensions
            corrupted = corrupted.view_as(images)  # Shape: (B, 3, 128, 128)
        else:
            corrupted = images


        # Reshape back to (batch, 3, D, D)
        corrupted = corrupted.view(B, N, D, D)

        ctf_params = ctf_params.view(B, N, 9)    

        return corrupted, ctf_params, obj_ids
