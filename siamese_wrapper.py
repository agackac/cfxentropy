class Siamese(Model):
    def __init__(self, FE, naive=False, cutoff=10, center=False, raw=False, add_loss=False):
        super(Siamese,self).__init__()
        
        self.FE = FE              # the feature extractor model 
        self.center = center      # center predictions against the batch-level mean before computing probabilities (re-calibration) 
        self.raw = raw            # whether counterfactuals should be computed over raw linear predictions instead of hyperbolic tangent
        self.naive = naive        # whether to use naive training (based on dichotomized PHQ-8 labels) instead of counterfactual training 
        self.cutoff = cutoff      # PHQ-8 cutoff for generating naive classification labels
        self.add_loss = add_loss  # whether to add auxillary loss from forecasting layer (multitask experiment)
        
        self.bce = BinaryCrossentropy(from_logits=False)
        self.loss_tracker = Mean(name='loss')
        
        self.iAUC = AUC(from_logits=False) # AUC for within-subject discrimination of counterfactual pairs
        self.nAUC = AUC(from_logits=True)  # AUC for naive classification of counterfactual pairs
        self.xAUC = AUC(from_logits=True)  # AUC for cross-sectional discrimination of counterfactual pairs
        self.rAUC = AUC(from_logits=True)  # AUC for naive classification of random sample (i.i.d.)
        self.iacc = BinaryAccuracy()       # accuracy for which counterfactual is more depressed 
        self.icorr = Mean()                # longitudinal correlation of predictions with severity (delta predictions x delta PHQ-8 sum score)
        self.ncorr = Mean()                # cross-sectional correlation of predictions with severity (absolute predictions x absolute PHQ-8 sum score)
        self.gcorr = Mean()                # cross-sectional correlation of predictions with gender
        self.igcorr = Mean()               # cross-sectional correlation of gender with longitudinal variability in predictions
        self.mu = Mean()                   # mean prediction (to track calibration under covariate shifts)
        self.diff = Mean()                 # standardized difference between pairs (counterfactual separation relative to cross-sectional variance)
        self.stdev = Mean()                # standard deviation of pairwise means (measure of cross-sectional variability)
        
        
        
    @property
    def metrics(self):
        return [self.loss_tracker,
            self.iAUC, self.xAUC,
            self.nAUC, self.rAUC,
            self.iacc, 
            self.icorr, self.ncorr, 
            self.gcorr, self.igcorr,
            self.mu,
            self.diff,
            self.stdev,
        ]
    
    def correlate(self, y, x, marginal=True, gender=False):
        corr = tfp.stats.correlation(y,x)
        if gender==False:
            if marginal==True:
                self.ncorr.update_state(corr)
            else:
                self.icorr.update_state(corr)
        else:
            if marginal==True:
                self.gcorr.update_state(corr)
            else:
                self.igcorr.update_state(corr)
                
    def compute_stats(self, x, eps=1e-3):
        batch_mean  = tf.reduce_mean(x)
        batch_stdev = tf.math.reduce_std(x)
        
        if self.center==True:
            x -= batch_mean
            
        # retrieve counterfactual branches
        D,H = tf.split(x,2,axis=0)
        
        ### compute pairwise differences 
        raw_diff = D-H                             
        stn_diff = raw_diff / (batch_stdev + eps)  # standardized difference
        
        # concatenate counterfactual paiars
        cf      = tf.concat([D,H],-1)
        
        # compute pairwise means & standard deviation
        id_mean = tf.reduce_mean(cf,-1)
        id_std  = tf.math.reduce_std(id_mean)
        
        # update calibration & separation statistics
        self.mu.update_state(batch_mean)
        self.stdev.update_state(id_std)
        self.diff.update_state(stn_diff)
        
        return x, cf, raw_diff
        
    def get_cf(self, inputs):  # counterfactual training pipeline
            
        D,H, gender = inputs
            
        # unpack counterfactual branches (spectrogram & PHQ-8 total)
        D,Dy = D     # depressed
        H,Hy = H     # healthy
        x = [D, H]   # spectrograms
        y = [Dy, Hy] # PHQ-8 total scores
        
        # concatenate raw PHQ-8 scores
        phq_abs = tf.concat(y,0)
        # compute dichotomized labels
        phq_cut = tf.cast(tf.greater_equal(phq_abs,self.cutoff),tf.float32)
        # compute counterfactual labels
        cf_y    = tf.concat([tf.ones_like(Dy), tf.zeros_like(Hy)], axis = -1) # within-pair
        xs_y    = tf.concat([tf.ones_like(Dy), tf.zeros_like(Hy)], axis = 0)  # cross-sectional
        
        # get model predictions (raw, linear) 
        x = tf.concat(x,0)
        x = self.FE(x)
        
        x, cf, raw_diff = self.compute_stats(x)
        
        # compute probabilities from raw predictions
        if self.raw == False:
            cf = tf.nn.tanh(cf)
        cf = tf.nn.softmax(cf,axis=-1)   # counterfactual
        xs = tf.nn.sigmoid(x)            # cross-sectional
        
        
        # correlate pairwise differences in predictions with... 
        
        ## differences in PHQ-8 total  
        phq_diff = Dy - Hy
        self.correlate(phq_diff, raw_diff, marginal=False)
        
        ## participant's gender
        self.correlate(gender, raw_diff, marginal=False, gender=True)
        
        # correlate predicted probabilities with gender, cross-sectionally
        gender = tf.concat([gender,gender],0)
        self.correlate(gender,x, gender=True)
        
        # update AUC metrics
        self.iAUC.update_state(cf_y, cf)
        self.iacc.update_state(cf_y, cf)
        self.xAUC.update_state(xs_y, xs)
        self.nAUC.update_state(phq_cut, xs)
        
        # compute counterfactual crossentropy loss
        loss = self.bce(cf_y,cf)
            
        return loss

    def get_naive(self, inputs):
        x,y = inputs
        x = self.FE(x)
        self.correlate(y,x)
        x = tf.nn.sigmoid(x)
        y = tf.cast(tf.greater_equal(y,self.cutoff),tf.float32)
        self.rAUC.update_state(y,x)
        loss = self.bce(y, x)
        return loss
        
    def train_step(self, inputs):
        
        with tf.GradientTape() as tape:
            loss = self.get_cf(inputs) if self.naive==False else self.get_naive(inputs) 
            if self.add_loss==True:
                loss += sum(self.FE.losses)
            
        self.loss_tracker.update_state(loss)
        trainables = self.FE.trainable_variables
        grads = tape.gradient(loss,trainables)
        self.optimizer.apply_gradients(zip(grads,trainables))
        
        if self.naive==True:
            return {
                'loss':self.loss_tracker.result(),
                'nAUC':self.rAUC.result(),
            }
        else:
            return {
                'loss':self.loss_tracker.result(),
                'nAUC':self.nAUC.result(),
                'cf_accuracy':self.iacc.result(),
            }
    
    def test_step(self, inputs):
        cf=True
        try:
            inputs, naive = inputs
            loss = self.get_cf(inputs)
            self.get_naive(naive)
        except:
            try:
                loss = self.get_cf(inputs)
            except:
                loss = self.get_naive(inputs)
                cf = False
        self.loss_tracker.update_state(loss)
        
        res= {
            'loss':self.loss_tracker.result(),
            'rAUC':self.rAUC.result(),
            'ncorr':self.ncorr.result(),
            'diff':self.diff.result(),
            'mean':self.mu.result(),
            'stdev':self.stdev.result(),
        }
        if cf==True:
            res.update(
           { 'cf_accuracy':self.iacc.result(),
            'iAUC':self.iAUC.result(),
            'nAUC':self.nAUC.result(),
            'xAUC':self.xAUC.result(),
            'icorr':self.icorr.result(),
            'gcorr':self.gcorr.result(),
            'igcorr':self.igcorr.result(),
           })
        return res
    
    def call(self, x):
        # returns batch mean as individual predictions use batches of the same recording to counteract variability across slices
        return tf.reduce_mean(self.FE(x)) 
