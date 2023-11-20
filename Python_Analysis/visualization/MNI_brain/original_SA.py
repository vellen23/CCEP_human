def combine_mni_and_zscore(df, start_time, end_time):
    mni_electrode_coords = []
    
    for _, row in df.iterrows():
        data = load_electrode_files(row['PAT'], row['STAGE'], row['LABEL'])
        avg_z_score = calculate_avg_zscore(data, start_time, end_time)
        mni_coords = [abs(row['MNI_X']), row['MNI_Y'], row['MNI_Z']]
        mni_electrode_coords.append((mni_coords, avg_z_score))
        
    return mni_electrode_coords


x = ActivationMap(radius=10)


from matplotlib.animation import FuncAnimation

fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
i = 0

frames_data = []
for frame in np.arange(900, 2048, 5):
    print(frame)
    mni_and_zscores = combine_mni_and_zscore(filtered_data, frame, frame + 10, DATA_DIR)
    frames_data.append(mni_and_zscores)

# Instead of threshold, have option to set all values below a value to 0

def update(frame):
    print(f'{frame}/{len(frames_data)}')
    ax.cla()  # Clear the current axis to plot the new data
    x.set_data(frames_data[frame])
    x.plot_3d(threshold=0, ax=ax, colorbar=False, vmax=40)

    time_in_ms = int(-100 + frame * 5 * (1000/1024))
    ax.text2D(0.05, 0.95, f"{time_in_ms} ms", transform=ax.transAxes, fontsize=10)

print('111')
ani = FuncAnimation(fig, update, frames=len(frames_data), repeat=False)
ani.save('./animation_nrem.mp4', writer='ffmpeg', fps=24)
ani.save('./animation_nrem.gif', writer='imagemagick', fps=24)



import numpy as np
import nibabel as nib
from nilearn import datasets, plotting, surface
import os

class ActivationMap:

    def __init__(self, mni_activation_data=None, radius=5, threshold=None):

        os.environ['QT_API'] = 'pyside2'
        os.environ['ETS_TOOLKIT'] = 'qt4'

        mni_template = datasets.fetch_icbm152_2009()
        template_img = nib.load(mni_template['t1'])
        template_data = template_img.get_fdata()
        
        self.fsaverage = datasets.fetch_surf_fsaverage(mesh='fsaverage5')
        self.affine = template_img.affine
        self.template_img = template_img
        self.template_data = template_data
        self.threshold = threshold
        
        self.mni_activation_data = mni_activation_data
        self.radius = radius

        self.voxel_activation_data = None
        self.activation_map = None
        self.activation_img = None
        self.activation_on_surface = None

        if mni_activation_data is not None:
            self.set_data(mni_activation_data)

    def _mni_to_voxel(self, mni_coords, activation):
        voxel_coords = np.round(np.linalg.inv(self.affine) @ np.append(mni_coords, 1))[:3]
        voxel_coords[np.isnan(voxel_coords)] = 0
        return [[int(v) for v in voxel_coords], activation]

    def _parse_electrodes(self):
        return [self._mni_to_voxel(coord, activation) for coord, activation in self.mni_activation_data]

    def _weight_function(self, i, j, k):
        distance = abs(i) + abs(j) + abs(k)
        max_distance = 3 * self.radius
        weight = 1 - (distance / max_distance)
        return weight
    
    def _make_activation_map(self):
        activation_map = np.zeros_like(self.template_data)
        for xyz, value in self.voxel_activation_data:
            x, y, z = xyz
            for i in range(-self.radius, self.radius+1):
                for j in range(-self.radius, self.radius+1):
                    for k in range(-self.radius, self.radius+1):
                        if i**2 + j**2 + k**2 <= self.radius**2:
                            activation_map[x+k, y+j, z+i] += value * self._weight_function(i, j, k)
        return activation_map
    
    def _make_activation_img(self):
        return nib.Nifti1Image(self.activation_map,
                               affine=self.template_img.affine)
    
    def _make_surface_activation(self):
        activation_on_surface = surface.vol_to_surf(self.activation_img, self.fsaverage.pial_right)
        nan_mask = np.isnan(activation_on_surface)
        inf_mask = np.isinf(activation_on_surface)
        activation_on_surface[nan_mask] = 0
        activation_on_surface[inf_mask] = 0
        return activation_on_surface
    
    def set_data(self, mni_activation_data):
        self.mni_activation_data = mni_activation_data
        self.voxel_activation_data = self._parse_electrodes()
        self.activation_map = self._make_activation_map()
        self.activation_img = self._make_activation_img()
        self.activation_on_surface = self._make_surface_activation()

    def plot_2d(self, threshold=None):
        threshold = threshold if not threshold is None else self.threshold
        plotting.plot_stat_map(self.activation_img, threshold=0.1)
        plotting.show()

    def plot_3d(self, threshold=None, ax=None, colorbar=True, vmax=None, cmap=None):
        threshold = threshold if not threshold is None else self.threshold
        plotting.plot_surf_stat_map(self.fsaverage.infl_right,#white_right,#infl_right
                                    self.activation_on_surface,
                                    hemi='right',
                                    threshold=threshold,
                                    alpha=0.9,
                                    vmax=vmax,
                                    #cmap=cmap,
                                    bg_map=self.fsaverage.sulc_right,
                                    bg_on_data=True,
                                    colorbar=colorbar,
                                    symmetric_cbar=False,
                                    axes=ax)
        if ax is not None:
            plotting.show()
