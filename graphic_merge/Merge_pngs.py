import numpy as np
import PIL
from PIL import Image

list_im =  ['CDW_sim_HK0L0=-35-6.jpg', 'compil_CDW2.jpg']# ['DAC_gasket_diamond.jpg', 'Marein_0GPa_0KL.jpg', 'Marein_5GPa_0KL.jpg'] # ['image106.jpg', 'image104.jpg', 'image105.jpg']
imgs    = [ Image.open(i) for i in list_im ]
# pick the image which is the smallest, and resize the others to match it (can be arbitrary image shape here)
min_shape = sorted( [(np.sum(i.size), i.size ) for i in imgs])[0][1]
imgs_comb = np.hstack( (np.asarray( i.resize(min_shape) ) for i in imgs ) )

# save that beautiful picture
# imgs_comb = Image.fromarray( imgs_comb)
# imgs_comb.save( 'CDW_compil2.jpg' )  #DAC_gasket_collapse_shattering

"""for a vertical stacking it is simple: use vstack
"""
imgs_comb = np.vstack( (np.asarray( i.resize(min_shape) ) for i in imgs ) )
imgs_comb = Image.fromarray( imgs_comb)
imgs_comb.save( 'CDW_compil2.jpg' )