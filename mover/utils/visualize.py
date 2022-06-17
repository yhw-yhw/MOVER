import numpy as np
from mover.constants import NYU40CLASSES
import seaborn as sns
nyu_colorbox = np.array(sns.color_palette("hls", n_colors=len(NYU40CLASSES)))

nyu_color_palette = dict()
for class_id, color in enumerate(nyu_colorbox):
    nyu_color_palette[class_id] = color.tolist()

body_color= [255 / 255.0, 99/255.0, 71/ 255.0] #[0.65098039, 0.74117647, 0.85882353]


define_color_map = {4: [255 / 255.0, 153 / 255.0, 0 / 255.0],
                    5: [255 / 255.0, 215 / 255.0, 0 / 255.0],
                    6: [30 / 255.0, 144 / 255.0, 255 / 255.0],
                    7: [255/ 255.0,20/ 255.0,147/ 255.0], 
}
