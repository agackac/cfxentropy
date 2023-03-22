class MismatchLayer(Layer):
    def __init__(self, weight, shift, l2_normalize=True, stop_gradient=False, out='concat'):
        super(MismatchLayer,self).__init__()
        self.w = weight
        self.shift = shift
        self.out = out
        self.l2_norm = l2_normalize
        self.stop_grad = stop_gradient
        
    def call(self, pred, true, context):
        if self.stop_grad==True:
            true = tf.stop_gradient(true)
        if self.l2_norm==True:
            pred = K.l2_normalize(pred,-1)
            true = K.l2_normalize(true,-1)
        true = tf.roll(true,-self.shift,axis=1)
        loss = tf.reduce_sum(tf.square(pred-true),axis=-1, keepdims=True)
        mask = tf.concat([tf.ones_like(loss[:,self.shift:,:]), tf.zeros_like(loss[:,:self.shift,:])],axis=1)
        loss *= mask
        if self.out=='concat':
            out = tf.concat([loss, context], axis=-1)
        elif self.out=='loss':
            out = loss
        else:
            out = context
        self.add_loss(tf.reduce_mean(loss*self.w))
        return out
    
    

def multitask_ESN(duration=500, bn=False, rnn_dims=128, radius=.2, kernel_size=5, maxpool=50, mlp_units=[64], use_mask=False, dropout=.5, nfreq=0, ntime=0, auxillary_loss_weight=0, filters=None, context_afn='softmax', multitask_output='concat', shift=25, proj=None):
    
    inputs = Input((duration,40))
    if use_mask==True:  # set to False by default as all recordings are above 5s duration - set to True for longer duration runs
        masking = Masking(0.)
        mask = masking.compute_mask(inputs)
        x = masking(inputs)
    else:
        mask = None
        x = inputs
        
    if (nfreq+ntime)>0:
        x = SpecAugment(freq_mask_param=nfreq, time_mask_param=ntime, n_freq_mask=nfreq, n_time_mask=ntime)(x)
        
    x = UnitNormalization()(x) if bn==False else BatchNormalization()(x)
    x = tfa.layers.ESN(rnn_dims, spectral_radius=radius, return_sequences=True)(x) 
    x = DepthwiseConv1D(kernel_size, padding='same')(x)
    x = PReLU('ones',shared_axes=[1])(x)
    
    if maxpool!=None:
        mp = MaxPooling1D(maxpool,1,padding='same')(x)
        x = Concatenate()([x,mp])
        target = mp
    else:
        target = x
        
    if filters!=None:
        latents  = Conv1D(filters, kernel_size=1, padding='same', activation=context_afn)(x)
        forecast = Dense(rnn_dims, use_bias=False)(latents)
        context  = MismatchLayer(auxillary_loss_weight,shift,out=multitask_output)(pred=forecast, true=target, context=latents)
        if proj!=None:
            context = Dense(proj,'softmax')(context)
        x = Concatenate()([x, context])
        
    x = GlobalAveragePooling1D()(x,mask=mask)
    
    for units in mlp_units:
        x = Dropout(dropout)(x)
        x = Dense(units,'relu')(x)
        
    x = Dense(1)(x)
    
    FE = Model(inputs,x)
    
    return FE
