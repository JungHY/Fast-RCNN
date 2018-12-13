import keras

class RoIPoolingConv(keras.engine.topology.Layer) :
    """
    ROI Pooling layer for 2D inputs
    pool_size : int
    pooling region의 크기. 7일 경우 7 x 7 크기의 출력

    num_rois : roi의 개수
    
    input_shape
    img(1, channels, rows, cols) or img(1, rows, clos, cahnnels)
    roi(1, rum_rois, 4) = (x, y, w, h)
    """     

    def __init__(self, pool_size, num_rois, **kwargs) :
        self.dim_ordering = keras.backend.image_dim_ordering()
        
        self.pool_size = pool_size
        self.num_rois = num_rois

        super(RoIPoolingConv, self).__init__(**kwargs)

    def build(self, input_shape) :
        self.nb_channels = input_shape[0][3]

    def compute_output_shape(self, input_shape):
        return None, self.num_rois, self.pool_size, self.pool_size, self.nb_channels

    def call(self, x, mask=None) :

        assert(len(x) == 2)

        img = x[0]
        rois = x[1]

        input_shape = keras.backend.shape(img)

        outputs = []

        for roi_idx in range(self.num_rois) :

            x = rois[0, roi_idx, 0]
            y = rois[0, roi_idx, 1]
            w = rois[0, roi_idx, 2]
            h = rois[0, roi_idx, 3]

            row_length = w / float(self.pool_size)
            col_length = h / float(self.pool_size)

            num_pool_regions = self.pool_size

            x = keras.backend.cast(x, 'int32')
            y = keras.backend.cast(y, 'int32')
            w = keras.backend.cast(w, 'int32')
            h = keras.backend.cast(h, 'int32')

            rs = tf.image.resize_images(img[:, y:y+h, x:x+w, :], (self.pool_size, self.pool_size))
            outputs.append(rs)

        final_output = keras.backend.concatenate(outputs, axis=0)
        final_output = keras.backend.reshape(final_output, (1, self.num_rois, self.pool_size, self.pool_size, self.nb_channels))

        final_output = keras.backend.permute_dimensions(final_output, (0,1,2,3,4))

        return final_output