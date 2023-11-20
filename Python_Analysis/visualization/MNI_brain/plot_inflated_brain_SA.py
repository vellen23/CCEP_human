import os
import numpy as np
import nibabel as nib
from nilearn import datasets, plotting, surface
import matplotlib.pyplot as plt
import pandas as pd
sub_path = 'X:\\4 e-Lab\\'  # y:\\eLab
class ActivationMap:

    def __init__(self, mni_activation_data=None, path_FS=None, radius=5, threshold=None, hemi='right'):
        # Set the environment variables
        os.environ['QT_API'] = 'pyside2'
        os.environ['ETS_TOOLKIT'] = 'qt4'

        # Decide whether to use MNI template or the patient's MRI
        if path_FS is None:
            mni_template = datasets.fetch_icbm152_2009()
            template_img = nib.load(mni_template['t1'])
            self.in_mni_space = True
        else:
            t1_path = os.path.join(path_FS, "mri", "T1.mgz")
            template_img = nib.load(t1_path)
            self.in_mni_space = False

        self.template_data = template_img.get_fdata()
        self.affine = template_img.affine

        # Load surface data if in MNI space, else load patient-specific data
        if self.in_mni_space:
            self.fsaverage = datasets.fetch_surf_fsaverage(mesh='fsaverage5')
        else:
            if hemi == 'right':
                self.pial = os.path.join(path_FS, "surf", "rh.pial")
                self.sulc_right = os.path.join(path_FS, "surf", "rh.sulc")
            else:
                self.pial = os.path.join(path_FS, "surf", "lh.pial")
                self.sulc = os.path.join(path_FS, "surf", "lh.sulc")

        # Initialize other attributes
        self.mni_activation_data = mni_activation_data
        self.radius = radius
        self.threshold = threshold if threshold is not None else 0
        self.hemi = hemi
        # Compute and set data if mni_activation_data is given
        if mni_activation_data is not None:
            self.set_data(mni_activation_data)

    def _mni_to_voxel(self, mni_coords, activation):
        """Convert MNI coordinates to voxel coordinates."""
        voxel_coords = np.round(np.linalg.inv(self.affine) @ np.append(mni_coords, 1))[:3]
        voxel_coords[np.isnan(voxel_coords)] = 0
        return [list(map(int, voxel_coords)), activation]

    def _parse_electrodes(self):
        """Parse electrode data."""
        return [self._mni_to_voxel(coord, activation) for coord, activation in self.mni_activation_data]

    def _weight_function(self, i, j, k):
        """Compute a weight based on distance."""
        distance = abs(i) + abs(j) + abs(k)
        max_distance = 3 * self.radius
        return 1 - (distance / max_distance)

    def _make_activation_map(self):
        """Create an activation map from voxel data."""
        activation_map = np.zeros_like(self.template_data)
        for (x, y, z), value in self.voxel_activation_data:
            for i in range(-self.radius, self.radius + 1):
                for j in range(-self.radius, self.radius + 1):
                    for k in range(-self.radius, self.radius + 1):
                        if i ** 2 + j ** 2 + k ** 2 <= self.radius ** 2:
                            activation_map[x + k, y + j, z + i] += value * self._weight_function(i, j, k)
        return activation_map

    def _make_surface_activation(self):
        """Interpolate volume data onto a surface mesh."""
        if self.in_mni_space:
            mesh = self.fsaverage.pial_right
        else:
            mesh = self.pial

        try:
            return surface.vol_to_surf(self.activation_img, mesh)
        except Exception as e:
            print(f"Error during volume to surface conversion: {e}")
            return None

    def set_data(self, mni_activation_data):
        """Set the activation data and compute the associated maps."""
        self.mni_activation_data = mni_activation_data
        self.voxel_activation_data = self._parse_electrodes()
        self.activation_map = self._make_activation_map()
        self.activation_img = nib.Nifti1Image(self.activation_map, affine=self.affine)
        self.activation_on_surface = self._make_surface_activation()

    def plot_2d(self, threshold=None):
        """Plot a 2D activation map."""
        threshold = threshold or self.threshold
        plotting.plot_stat_map(self.activation_img, threshold=threshold)
        plotting.show()

    def plot_3d(self, threshold=None, ax=None, colorbar=True, vmax=None, cmap=None):
        """Plot a 3D activation map."""
        threshold = threshold or self.threshold
        if self.in_mni_space:
            # self.hemi , self.fsaverage.infl_right, self.fsaverage.sulc_right

            plotting.plot_surf_stat_map(self.fsaverage['infl_' + self.hemi], self.activation_on_surface, hemi=self.hemi,
                                        threshold=threshold, alpha=0.9, vmax=vmax,
                                        bg_map=self.fsaverage['sulc_' + self.hemi],
                                        bg_on_data=True, colorbar=colorbar, symmetric_cbar=False, axes=ax)
        else:
            plotting.plot_surf_stat_map(self.pial, self.activation_on_surface, hemi=self.hemi,
                                        threshold=threshold, alpha=0.9, vmax=vmax, bg_map=self.sulc,
                                        bg_on_data=True, colorbar=colorbar, symmetric_cbar=False, axes=ax)
        # if ax:
        #    plotting.show()


# Example data - this needs to be actual MNI coordinates and activation values
# sample_mni_activation_data = [([15, 0, 7], 10), ([7, 3, 10], 5)]  # replace with actual data
sc = 19
subj = 'EL011'

path_FS = os.path.join(sub_path, 'Patients', subj, 'Imaging', 'Reconstruction', 'Fs')
path_patient_analysis = os.path.join(sub_path, 'EvM', 'Projects', 'EL_experiment', 'Analysis', 'Patients', subj)

lbls = pd.read_excel(os.path.join(sub_path, 'Patients', subj, 'Electrodes', subj + "_labels.xlsx"), header=0, sheet_name='BP')
if "type" in lbls.columns:
    lbls = lbls[lbls.type=='SEEG']
    lbls = lbls.reset_index(drop=True)
coord = lbls[['x', 'y', 'z']].values
labels_all = lbls.label.values
path_save = os.path.join(path_patient_analysis, 'BrainMapping', 'CR', 'Visualization', 'inf_brain', labels_all[sc])
LL_time = np.load(os.path.join(path_save, 'LL.npy'))
# load data

bad = np.where(np.isnan(np.nanmean(LL_time,1)))[0]
LL_time = np.delete(LL_time, bad, 0)
coord = np.delete(coord, bad, 0)
# Instantiate ActivationMap with the sample data
# x = ActivationMap(mni_activation_data=sample_mni_activation_data, path_FS=path_FS, radius=10, hemi = 'left')
x = ActivationMap(path_FS=path_FS, radius=10, hemi = 'left')
# Ensure that set_data is called (this should already be done in the constructor, but just to make sure)
# x.set_data()

# Now, attempt to plot
# x.plot_2d(threshold=0)

# Test whether 3D plot is possible
#fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
#x.plot_3d(threshold=0, ax=ax, colorbar=False, vmax=40)
# Number of channels
channels = coord.shape[0]
# Loop over each timepoint
tp =np.arange(40, 161, 5)
for timepoint in tp:
    # Create the activation data for the given timepoint
    mni_activation_data = [(list(coord[i]), LL_time[i][timepoint]) for i in range(channels)]
    # Set the data and plot
    x.set_data(mni_activation_data)
    fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
    x.plot_3d(threshold=0, ax=ax, colorbar=False, vmax=5)
    # Save the figure
    plt.savefig(os.path.join(path_save, 'figures', f"frame_{timepoint}.png"))
    plt.close(fig)

print('Done')

# After looping, compile the frames to GIF
import imageio

# Create a GIF from the saved frames
frames = [imageio.imread(os.path.join(path_save, 'figures', f"frame_{i}.png")) for i in tp]
imageio.mimsave(os.path.join(path_save, 'figures','activation_over_time.gif'), frames, duration=0.5)

print('Done')
