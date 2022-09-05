import numpy as np
import PIL
from PIL import Image

list_im = ['ac.jpg', 'd11(2).jpg', 'ztheta.jpg']# ['DAC_EuGa4_structure_1.jpg', 'DAC_EuGa4_structure_2.jpg', 'DAC_EuGa4_structure_3.jpg']# ['M(1)M(2).jpg', 'AlGa_planes.jpg']# ['stavinoah1_euga2al2_1.jpg', 'stavinoah1_euga2al2_2.jpg', 'stavinoah1_euga2al2_3.jpg'] #['stavinoah1_euga4_1.jpg', 'stavinoah1_euga4_2.jpg', 'stavinoah1_euga4_3.jpg']
imgs    = [ Image.open(i) for i in list_im ]
# pick the image which is the smallest, and resize the others to match it (can be arbitrary image shape here)
min_shape = sorted( [(np.sum(i.size), i.size ) for i in imgs])[0][1]
imgs_comb = np.hstack( (np.asarray( i.resize(min_shape) ) for i in imgs ) )

# save that beautiful picture
imgs_comb = Image.fromarray( imgs_comb)
imgs_comb.save( 'euga2al2_structure_overview.jpg.jpg' ) #stavinoha_euga2al2 #DAC_EuGa4_structure

"""for a vertical stacking it is simple: use vstack
"""
# imgs_comb = np.vstack( (np.asarray( i.resize(min_shape) ) for i in imgs ) )
# imgs_comb = Image.fromarray( imgs_comb)
# imgs_comb.save( 'euga2al2_structure_overview.jpg' )