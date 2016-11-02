from __future__ import division
from itertools import izip

import numpy as np
import numpy.random as rng
import theano
import theano.tensor as tt
import matplotlib.pyplot as plt

import NeuralNet as nn
import pdf
import helper


dtype = theano.config.floatX
isposint = lambda t: isinstance(t, int) and t > 0


class MDN:
    """Implements a mixture density network with full precision matrices."""

    def __init__(self, n_inputs, n_hiddens, act_fun, n_outputs, n_components):
        """
        Constructs an mdn with a given architecture. Note that the mdn has full precision matrices.
        :param n_inputs: dimensionality of the input
        :param n_hiddens: list with number of hidden units in the net
        :param act_fun: activation function type to use in the net
        :param n_outputs: dimensionality of the output
        :param n_components: number of mixture components
        :return: None
        """

        # check if inputs are of the right type
        assert isposint(n_inputs), 'Number of inputs must be a positive integer.'
        assert isposint(n_outputs), 'Number of outputs must be a positive integer.'
        assert isposint(n_components), 'Number of components must be a positive integer.'
        assert isinstance(n_hiddens, list), 'Number of hidden units must be a list of positive integers.'
        for h in n_hiddens:
            assert isposint(h), 'Number of hidden units must be a list of positive integers.'
        assert act_fun in ['logistic', 'tanh', 'linear', 'relu', 'softplus'], 'Unsupported activation function.'

        # construct the net
        self.net = nn.NeuralNet(n_inputs)
        for h in n_hiddens:
            self.net.addLayer(h, act_fun)
        self.input = self.net.hs[0]

        # mixing coefficients
        self.Wa = theano.shared((rng.randn(self.net.n_outputs, n_components) / np.sqrt(self.net.n_outputs + 1)).astype(dtype), name='Wa')
        self.ba = theano.shared(rng.randn(n_components).astype(dtype), name='ba')
        self.a = tt.nnet.softmax(tt.dot(self.net.hs[-1], self.Wa) + self.ba)

        # mixture means
        # the mean of each component is calculated separately. consider vectorizing this
        self.Wms = [theano.shared((rng.randn(self.net.n_outputs, n_outputs) / np.sqrt(self.net.n_outputs + 1)).astype(dtype), name='Wm'+str(i)) for i in xrange(n_components)]
        self.bms = [theano.shared(rng.randn(n_outputs).astype(dtype), name='bm'+str(i)) for i in xrange(n_components)]
        self.ms = [tt.dot(self.net.hs[-1], Wm) + bm for Wm, bm in izip(self.Wms, self.bms)]

        # mixture precisions
        # note that U here is an upper triangular matrix such that U'*U + V'*V is the precision
        self.WUs = [theano.shared((rng.randn(self.net.n_outputs, n_outputs**2) / np.sqrt(self.net.n_outputs + 1)).astype(dtype), name='WU'+str(i)) for i in xrange(n_components)]
        self.bUs = [theano.shared(rng.randn(n_outputs**2).astype(dtype), name='bU'+str(i)) for i in xrange(n_components)]
        aUs = [tt.reshape(tt.dot(self.net.hs[-1], WU) + bU, [-1, n_outputs, n_outputs]) for WU, bU in izip(self.WUs, self.bUs)]
        triu_mask = np.triu(np.ones([n_outputs, n_outputs], dtype=dtype), 1)
        diag_mask = np.eye(n_outputs, dtype=dtype)
        
        #self.Us = [triu_mask * aU + diag_mask * tt.exp(diag_mask * aU) for aU in aUs]
        self.Us = [triu_mask * aU + diag_mask * tt.exp(diag_mask * aU) for aU in aUs]
        ldetUs = [tt.sum(tt.sum(diag_mask * aU, axis=2), axis=1) for aU in aUs]
        self.Vs = [np.zeros((self.net.n_outputs, n_outputs, n_outputs)).astype(dtype) for i in xrange(n_components)]
        ldetUVs = [tt.zeros(self.net.n_outputs) for i in xrange(n_components)]
        for i in xrange(n_components):
            for j in xrange(self.net.n_outputs):
                U = self.Us[i][j,:,:]
                V = self.Vs[i][j,:,:]
                ldetUVs[i] = .5 * tt.log(tt.nlinalg.Det()( tt.dot(U.dimshuffle([1, 0]),U)  + V.T.dot(V)))

        # log probabilities
        self.y = tt.matrix('y')
        lprobs_comps_U = [-0.5 * tt.sum(tt.sum((self.y-m).dimshuffle([0, 'x', 1]) * U, axis=2)**2, axis=1) for m, U in izip(self.ms, self.Us)]
        lprobs_comps_V = [-0.5 * tt.sum(tt.sum((self.y-m).dimshuffle([0, 'x', 1]) * V, axis=2)**2, axis=1) for m, V in izip(self.ms, self.Vs)]
        lprobs_comps = [lU + lV + ldetUV for lU,lV,ldetUV in izip(lprobs_comps_U,lprobs_comps_V,ldetUVs)]        
        #lprobs_comps = [-0.5 * tt.sum(tt.sum((self.y-m).dimshuffle([0, 'x', 1]) * U, axis=2)**2, axis=1) + ldetU for m, U, ldetU in izip(self.ms, self.Us, ldetUs)]
        self.lprobs = tt.log(tt.sum(tt.exp(tt.stack(lprobs_comps, axis=1) + tt.log(self.a)), axis=1)) - (0.5 * n_outputs * np.log(2*np.pi))
        self.mlprob = -tt.mean(self.lprobs)

        # all parameters in one container
        self.parms = self.net.parms + [self.Wa, self.ba] + self.Wms + self.bms + self.WUs + self.bUs

        # theano evaluation functions, will be compiled when first needed
        self.eval_comps_f = None
        self.eval_lprobs_f = None

        # save these for later
        self.n_inputs = self.net.n_inputs
        self.n_outputs = n_outputs
        self.n_components = n_components
        self.act_fun = act_fun

    def set_sqrt_prior_precisions(self,Vs):
        assert len(Vs) == len(self.Us)
        self.Vs = Vs.copy()

    def initialize_mog(self, y):

        n_data, n_dim = y.shape
        assert n_dim == self.n_outputs

        # calculate mean and covariance from data
        m = np.mean(y, axis=0)
        S = np.dot(y.T, y) / n_data - np.outer(m, m)
        P = np.linalg.inv(S)
        U = np.linalg.cholesky(P).T

        # initialize mixing coefficients approx uniformly
        self.Wa.set_value((rng.randn(self.net.n_outputs, self.n_components) / np.sqrt(self.net.n_outputs + 1)).astype(dtype))
        self.ba.set_value(np.zeros(self.n_components, dtype=dtype))

        # initialize means approx with the data means
        for Wm, bm in izip(self.Wms, self.bms):
            Wm.set_value((rng.randn(self.net.n_outputs, self.n_outputs) / np.sqrt(self.net.n_outputs + 1)).astype(dtype))
            bm.set_value(m.astype(dtype) + 0.1 * rng.randn(self.n_outputs).astype(dtype))

        # initialize precisions with the data precisions
        diag_mask = np.eye(n_dim, dtype=bool)
        U[diag_mask] = np.log(U[diag_mask])
        for WU, bU in izip(self.WUs, self.bUs):
            WU.set_value((rng.randn(self.net.n_outputs, self.n_outputs**2) / np.sqrt(self.net.n_outputs + 1)).astype(dtype))
            bU.set_value(U.flatten().astype(dtype))


    def eval_comps(self, x):
        """
        Evaluate the parameters of all mixture components at given input locations.
        :param x: rows are input locations
        :return: mixing coefficients, means and scale matrices
        """

        # compile theano function, if haven't already done so
        if self.eval_comps_f == None:
            self.eval_comps_f = theano.function(
                inputs=[self.input],
                outputs=[self.a] + self.ms + self.Us
            )

        comps = self.eval_comps_f(x.astype(dtype))

        return comps[0], comps[1:self.n_components+1], comps[self.n_components+1:]


    def eval(self, xy):
        """
        Evaluate log probabilities for given input-output pairs.
        :param xy: a pair (x, y) where x rows are inputs and y rows are outputs
        :return: log probabilities: log p(y|x)
        """

        # compile theano function, if haven't already done so
        if self.eval_lprobs_f == None:
            self.eval_lprobs_f = theano.function(
                inputs=[self.input, self.y],
                outputs=self.lprobs
            )

        x, y = xy
        return self.eval_lprobs_f(x.astype(dtype), y.astype(dtype))


    def get_mog(self, x):
        """
        Return the conditional mog at location x.
        :param x: single input location
        :return: conditional mog at x
        """

        x = x[np.newaxis, :] if x.ndim == 1 else x
        assert x.shape[0] == 1

        # gather mog parameters
        a, ms, Us = self.eval_comps(x)
        a = a[0]
        ms = [m[0] for m in ms]
        Us = [U[0] for U in Us]

        # return mog
        return pdf.MoG(a=a, ms=ms, Us=Us)


    def gen(self, x, n_samples=1):
        """
        Generate samplers from the mdn conditioned on x.
        :param x: input vector
        :param n_samples: number of samples
        :return: samples
        """

        mog = self.get_mog(x)
        samples = mog.gen(n_samples)

        return samples


    def visualize_weights(self, layer, imsize, layout):
        """
        Displays the weights of a specified layer as images.
        :param layer: the layer whose weights to display
        :param imsize: the image size
        :param layout: number of rows and columns for each page
        :return: none
        """

        if layer < self.net.n_layers:
            self.net.visualize_weights(layer, imsize, layout)

        elif layer == self.net.n_layers:
            helper.disp_imdata(np.concatenate([W.get_value() for W in [self.Wa] + self.Wms + self.WUs], axis=1).T, imsize, layout)
            plt.show(block=False)

        else:
            raise ValueError('Layer {} doesn\'t exist.'.format(layer))


    def visualize_activations(self, x):
        """
        Visualizes the activations in the mdn caused by a given data minibatch.
        :param x: a minibatch of data
        :return: none
        """

        self.net.visualize_activations(x)

        forwprop = theano.function(
            inputs=[self.input],
            outputs=[self.a, tt.concatenate(self.ms, axis=1) + tt.concatenate([tt.reshape(U, [U.shape[0], -1]) for U in self.Us], axis=1)]
        )
        activations = forwprop(x.astype(dtype))

        for a, title in izip(activations, ['mixing coefficients', 'means', 'scale matrices']):

            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            ax.imshow(a, cmap='gray', interpolation='none')
            ax.set_title(title)
            ax.set_xlabel('layer units')
            ax.set_ylabel('data points')

        plt.show(block=False)


    def param_hist(self):
        """
        Displays a histogram of all parameters in the mdn.
        :return: none
        """

        self.net.param_hist()

        all_Wa = self.Wa.get_value().flatten()
        all_ba = self.ba.get_value().flatten()
        all_Wm = np.concatenate([Wm.get_value().flatten() for Wm in self.Wms])
        all_bm = np.concatenate([bm.get_value().flatten() for bm in self.bms])
        all_WU = np.concatenate([WU.get_value().flatten() for WU in self.WUs])
        all_bU = np.concatenate([bU.get_value().flatten() for bU in self.bUs])

        for W, b, title in izip([all_Wa, all_Wm, all_WU], [all_ba, all_bm, all_bU], ['mixing coefficients', 'means', 'scale matrices']):

            fig, (ax1, ax2) = plt.subplots(1, 2)

            nbins = int(np.sqrt(W.size))
            ax1.hist(W, nbins, normed=True)
            ax1.set_title('weights, ' + title)

            nbins = int(np.sqrt(b.size))
            ax2.hist(b, nbins, normed=True)
            ax2.set_title('biases, ' + title)

        plt.show(block=True)


class MDN_SVI:
    """
    Implements a bayesian mixture density network with full precision matrices, using stochastic variational inference.
    """

    def __init__(self, n_inputs, n_hiddens, act_fun, n_outputs, n_components, Vs = None):
        """
        Constructs an svi mdn with a given architecture. Note that the mdn has full precision matrices.
        :param n_inputs: dimensionality of the input
        :param n_hiddens: list with number of hidden units in the net
        :param act_fun: activation function type to use in the net
        :param n_outputs: dimensionality of the output
        :param n_components: number of mixture components
        :return: None
        """

        # check if inputs are of the right type
        assert isposint(n_inputs), 'Number of inputs must be a positive integer.'
        assert isposint(n_outputs), 'Number of outputs must be a positive integer.'
        assert isposint(n_components), 'Number of components must be a positive integer.'
        assert isinstance(n_hiddens, list), 'Number of hidden units must be a list of positive integers.'
        for h in n_hiddens:
            assert isposint(h), 'Number of hidden units must be a list of positive integers.'
        assert act_fun in ['logistic', 'tanh', 'linear', 'relu', 'softplus'], 'Unsupported activation function.'

        # construct the net
        self.net = nn.NeuralNetSvi(n_inputs)
        for h in n_hiddens:
            self.net.addLayer(h, act_fun)
        self.input = self.net.hs[0]
        self.srng = self.net.srng

        n_data = self.input.shape[0]

        if Vs is None:
            self.Vs = [theano.shared(np.zeros((n_outputs, n_outputs)).astype(dtype), name='V'+str(i)) for i in xrange(n_components)]
        else:
            self.Vs = [theano.shared(Vs[i].astype(dtype), name='V'+str(i)) for i in xrange(n_components)]

        # the naming scheme of the theano variables from now on might look a bit cryptic but it actually makes sense
        # each variable name has 3 or 4 letters, with the following meanings:
        # 1st letter: m=mean, s=variance, u=noise, z=random. note that s can also be the log std if convenient
        # 2nd letter: W=weights, b=biases or a=activations
        # 3rd letter: a=mixing coefficients, m=means, U=precisions
        # 4th letter: s if it's a list of variables, nothing otherwise
        # in general, capital means matrix, lowercase means vector(s)

        # mixing coefficients
        self.mWa = theano.shared((rng.randn(self.net.n_outputs, n_components) / np.sqrt(self.net.n_outputs + 1)).astype(dtype), name='mWa')
        self.mba = theano.shared(rng.randn(n_components).astype(dtype), name='mba')
        self.sWa = theano.shared(-5.0 * np.ones([self.net.n_outputs, n_components], dtype=dtype), name='sWa')
        self.sba = theano.shared(-5.0 * np.ones(n_components, dtype=dtype), name='sba')
        uaa = self.srng.normal((self.net.hs[-1].shape[0], n_components), dtype=dtype)
        maa = tt.dot(self.net.hs[-1], self.mWa) + self.mba
        saa = tt.dot(self.net.hs[-1]**2, tt.exp(2*self.sWa)) + tt.exp(2*self.sba)
        zaa = tt.sqrt(saa) * uaa + maa
        self.a = tt.nnet.softmax(zaa)

        # mixture means
        # the mean of each component is calculated separately. consider vectorizing this
        self.mWms = [theano.shared((rng.randn(self.net.n_outputs, n_outputs) / np.sqrt(self.net.n_outputs + 1)).astype(dtype), name='mWm'+str(i)) for i in xrange(n_components)]
        self.mbms = [theano.shared(rng.randn(n_outputs).astype(dtype), name='mbm'+str(i)) for i in xrange(n_components)]
        self.sWms = [theano.shared(-5.0 * np.ones([self.net.n_outputs, n_outputs], dtype=dtype), name='sWm'+str(i)) for i in xrange(n_components)]
        self.sbms = [theano.shared(-5.0 * np.ones(n_outputs, dtype=dtype), name='sbm'+str(i)) for i in xrange(n_components)]
        uams = [self.srng.normal((self.net.hs[-1].shape[0], n_outputs), dtype=dtype) for i in xrange(n_components)]
        mams = [tt.dot(self.net.hs[-1], mWm) + mbm for mWm, mbm in izip(self.mWms, self.mbms)]
        sams = [tt.dot(self.net.hs[-1]**2, tt.exp(2*sWm)) + tt.exp(2*sbm) for sWm, sbm in izip(self.sWms, self.sbms)]
        zams = [tt.sqrt(sam) * uam + mam for sam, uam, mam in izip(sams, uams, mams)]
        self.ms = zams

        # mixture precisions
        # note that U here is an upper triangular matrix such that U'*U + V'*V is the precision
        self.mWUs = [theano.shared((rng.randn(self.net.n_outputs, n_outputs**2) / np.sqrt(self.net.n_outputs + 1)).astype(dtype), name='mWU'+str(i)) for i in xrange(n_components)]
        self.mbUs = [theano.shared(rng.randn(n_outputs**2).astype(dtype), name='mbU'+str(i)) for i in xrange(n_components)]
        self.sWUs = [theano.shared(-5.0 * np.ones([self.net.n_outputs, n_outputs**2], dtype=dtype), name='sWU'+str(i)) for i in xrange(n_components)]
        self.sbUs = [theano.shared(-5.0 * np.ones(n_outputs**2, dtype=dtype), name='sbU'+str(i)) for i in xrange(n_components)]
        uaUs = [self.srng.normal((self.net.hs[-1].shape[0], n_outputs**2), dtype=dtype) for i in xrange(n_components)]
        maUs = [tt.dot(self.net.hs[-1], mWU) + mbU for mWU, mbU in izip(self.mWUs, self.mbUs)]
        saUs = [tt.dot(self.net.hs[-1]**2, tt.exp(2*sWU)) + tt.exp(2*sbU) for sWU, sbU in izip(self.sWUs, self.sbUs)]
        zaUs = [tt.sqrt(saU) * uaU + maU for saU, uaU, maU in izip(saUs, uaUs, maUs)]
        zaUs_reshaped = [tt.reshape(zaU, [-1, n_outputs, n_outputs]) for zaU in zaUs]
        triu_mask = np.triu(np.ones([n_outputs, n_outputs], dtype=dtype), 1)
        diag_mask = np.eye(n_outputs, dtype=dtype)
        self.Us = [triu_mask * zaU + diag_mask * tt.exp(diag_mask * zaU) for zaU in zaUs_reshaped]

        # computing logdet of precision matrices
        self.Ps = [theano.scan(fn=lambda U,V: tt.dot(U.dimshuffle([1, 0]),U) + tt.dot(V.dimshuffle([1, 0]),V),
                               outputs_info = None,
                               sequences = self.Us[i],
                               non_sequences = self.Vs[i] )[0] for i in xrange(n_components)]            
        ldetPs = [theano.scan(fn=lambda P: .5 * tt.log(tt.nlinalg.Det()( P )),
                              outputs_info = None,
                              sequences = self.Ps[i])[0] for i in xrange(n_components)]

        # log probabilities
        self.y = tt.matrix('y')
        self.lprobs_comps_U = [-0.5 * tt.sum(tt.sum((self.y-m).dimshuffle([0, 'x', 1]) * U, axis=2)**2, axis=1) for m, U in izip(self.ms, self.Us)]
        self.lprobs_comps_V = [-0.5 * tt.sum(tt.sum((self.y-m).dimshuffle([0, 'x', 1]) * V.dimshuffle(['x', 0, 1]), axis=2)**2, axis=1) for m, V in izip(self.ms, self.Vs)]
        lprobs_comps = [lU + lV + ldetP for lU,lV,ldetP in izip(self.lprobs_comps_U,self.lprobs_comps_V,ldetPs)]
        self.lprobs = tt.log(tt.sum(tt.exp(tt.stack(lprobs_comps, axis=1) + tt.log(self.a)), axis=1)) - (0.5 * n_outputs * np.log(2*np.pi))
        self.mlprob = -tt.mean(self.lprobs)

        # all parameters in one container
        self.uas = self.net.uas + [uaa] + uams + uaUs
        self.mas = self.net.mas + [maa] + mams + maUs
        self.zas = self.net.zas + [zaa] + zams + zaUs
        self.mps = self.net.mps + [self.mWa, self.mba] + self.mWms + self.mbms + self.mWUs + self.mbUs
        self.sps = self.net.sps + [self.sWa, self.sba] + self.sWms + self.sbms + self.sWUs + self.sbUs
        self.parms = self.mps + self.sps

        # theano evaluation functions, will be compiled when first needed
        self.eval_comps_f = None
        self.eval_lprobs_f = None
        self.eval_comps_f_rand = None
        self.eval_lprobs_f_rand = None

        # save these for later
        self.n_inputs = self.net.n_inputs
        self.n_outputs = n_outputs
        self.n_components = n_components
        self.act_fun = act_fun

    def set_sqrt_prior_precisions(self,Vs):
        assert len(Vs) == self.n_components

        for i in xrange(self.n_components):
            self.Vs[i].set_value(Vs[i].astype(dtype)) 

    def initialize_mog(self, y):

        n_data, n_dim = y.shape
        assert n_dim == self.n_outputs

        # calculate mean and covariance from data
        m = np.mean(y, axis=0)
        S = np.dot(y.T, y) / n_data - np.outer(m, m)
        P = np.linalg.inv(S)
        U = np.linalg.cholesky(P).T

        # initialize mixing coefficients approx uniformly
        self.mWa.set_value((rng.randn(self.net.n_outputs, self.n_components) / np.sqrt(self.net.n_outputs + 1)).astype(dtype))
        self.mba.set_value(np.zeros(self.n_components, dtype=dtype))

        # initialize means approx with the data means
        for mWm, mbm in izip(self.mWms, self.mbms):
            mWm.set_value((rng.randn(self.net.n_outputs, self.n_outputs) / np.sqrt(self.net.n_outputs + 1)).astype(dtype))
            mbm.set_value(m.astype(dtype) + 0.1 * rng.randn(self.n_outputs).astype(dtype))

        # initialize precisions with the data precisions
        diag_mask = np.eye(n_dim, dtype=bool)
        U[diag_mask] = np.log(U[diag_mask])
        for mWU, mbU in izip(self.mWUs, self.mbUs):
            mWU.set_value((rng.randn(self.net.n_outputs, self.n_outputs**2) / np.sqrt(self.net.n_outputs + 1)).astype(dtype))
            mbU.set_value(U.flatten().astype(dtype))


    def _create_constant_uas_across_datapoints(self):
        """
        Helper function. Creates and returns new theano variables representing noise, where noise is the same across
        datapoints in the minibatch. Useful for binding the original noise variables in evaluation function where
        randomness is required but same predictions are needed across minibatch.
        """

        n_data = tt.iscalar('n_data')

        net_uas = [tt.tile(self.srng.normal((n_units,), dtype=dtype), [n_data, 1]) for n_units in self.net.n_units[1:]]
        uaa = tt.tile(self.srng.normal((self.n_components,), dtype=dtype), [n_data, 1])
        uams = [tt.tile(self.srng.normal((self.n_outputs,), dtype=dtype), [n_data, 1]) for _ in xrange(self.n_components)]
        uaUs = [tt.tile(self.srng.normal((self.n_outputs**2,), dtype=dtype), [n_data, 1]) for _ in xrange(self.n_components)]

        # NOTE: order matters here
        uas = net_uas + [uaa] + uams + uaUs

        return n_data, uas


    def eval_comps(self, x, rand=False):
        """
        Evaluate the parameters of all mixture components at given input locations.
        :param x: rows are input locations
        :param rand: if True, inject randomness in the activations
        :return: mixing coefficients, means and scale matrices
        """

        if rand:

            # compile theano function, if haven't already done so
            if self.eval_comps_f_rand == None:

                n_data, uas = self._create_constant_uas_across_datapoints()

                self.eval_comps_f_rand = theano.function(
                    inputs=[self.input, n_data],
                    outputs=[self.a] + self.ms + self.Ps,
                    givens=zip(self.uas, uas)
                )

            comps = self.eval_comps_f_rand(x.astype(dtype), x.shape[0])

        else:

            # compile theano function, if haven't already done so
            if self.eval_comps_f == None:
                self.eval_comps_f = theano.function(
                    inputs=[self.input],
                    outputs=[self.a] + self.ms + self.Ps,
                    givens=zip(self.zas, self.mas)
                )

            comps = self.eval_comps_f(x.astype(dtype))

        return comps[0], comps[1:self.n_components+1], comps[self.n_components+1:]


    def eval(self, xy, rand=False):
        """
        Evaluate log probabilities for given input-output pairs.
        :param xy: a pair (x, y) where x rows are inputs and y rows are outputs
        :param rand: if True, inject randomness in the activations
        :return: log probabilities: log p(y|x)
        """

        x, y = xy
        assert x.shape[0] == y.shape[0]

        if rand:

            # compile theano function, if haven't already done so
            if self.eval_lprobs_f_rand == None:

                n_data, uas = self._create_constant_uas_across_datapoints()

                self.eval_lprobs_f_rand = theano.function(
                    inputs=[self.input, self.y, n_data],
                    outputs=self.lprobs,
                    givens=zip(self.uas, uas)
                )

            return self.eval_lprobs_f_rand(x.astype(dtype), y.astype(dtype), x.shape[0], self.Vs)

        else:

            # compile theano function, if haven't already done so
            if self.eval_lprobs_f == None:
                self.eval_lprobs_f = theano.function(
                    inputs=[self.input, self.y],
                    outputs=self.lprobs,
                    givens=zip(self.zas, self.mas)
                )

            return self.eval_lprobs_f(x.astype(dtype), y.astype(dtype))


    def eval_debug(self, xy):

        x, y = xy
        assert x.shape[0] == y.shape[0]        
        self.eval_comps_f_debug = theano.function(
            inputs=[self.input,self.y],
            outputs=[self.a] + self.ms + self.Us + self.Ps + [self.lprobs],
            givens=zip(self.zas, self.mas)
        )
        comps = self.eval_comps_f_debug(x.astype(dtype), y.astype(dtype))
        return comps[0], comps[1:self.n_components+1], comps[self.n_components+1:2*self.n_components+1], comps[2*self.n_components+1:3*self.n_components+1],  comps[-1]


    def get_mog(self, x, n_samples=None):
        """
        Return the conditional mog at location x. The mdn can either be used without randomness, or several samples from
        it can make up the mog.
        :param x: single input location
        :param n_samples: number of mdn samples to put in the mog; if None switch randomness off
        :return: conditional mog at x
        """

        # prepare input
        x = x[np.newaxis, :] if x.ndim == 1 else x
        assert x.shape[0] == 1

        if n_samples is None:  # no randomness

            # gather mog parameters
            a, ms, Ps = self.eval_comps(x, rand=False)
            a = a[0]
            ms = [m[0] for m in ms]
            Ps = [P[0] for P in Ps]

        else:  # sample from mdn, and form a mog with all the samples

            assert isposint(n_samples)

            a = np.array([])
            ms = []
            Us = []

            for _ in xrange(n_samples):

                # generate a mog and gather its parameters
                ai, mis, Pis = self.eval_comps(x, rand=True)
                a = np.append(a, ai[0] / n_samples)
                ms += [mi[0] for mi in mis]
                Ps += [Pi[0] for Pi in Pis]

        # return mog
        return pdf.MoG(a=a, ms=ms, Ps=Ps)


    def gen(self, x, n_samples=1, rand=False):
        """
        Generate samplers from the mdn conditioned on x.
        :param x: input vector
        :param n_samples: number of samples
        :param rand: if True, each sample is generated from a different mdn
        :return: samples
        """

        if rand:  # each sample from a different mdn

            samples = np.empty([n_samples, self.n_outputs])

            for i in xrange(n_samples):
                mog = self.get_mog(x, n_samples=1)
                samples[i] = mog.gen(n_samples=1)

        else:  # all samples from same mdn

            mog = self.get_mog(x, n_samples=None)
            samples = mog.gen(n_samples)

        return samples


    def turn_into_conventional(self):
        """
        Returns a conventional mdn that is otherwise the same as this mdn.
        """

        net = MDN(n_inputs=self.n_inputs, n_hiddens=self.net.n_units[1:], act_fun=self.act_fun, n_outputs=self.n_outputs, n_components=self.n_components, Vs = self.Vs)

        for W, b, mW, mb in izip(net.net.Ws, net.net.bs, self.net.mWs, self.net.mbs):
            W.set_value(mW.get_value())
            b.set_value(mb.get_value())

        net.Wa.set_value(self.mWa.get_value())
        net.ba.set_value(self.mba.get_value())

        for Wm, bm, WU, bU, mWm, mbm, mWU, mbU in izip(net.Wms, net.bms, net.WUs, net.bUs, self.mWms, self.mbms, self.mWUs, self.mbUs):
            Wm.set_value(mWm.get_value())
            bm.set_value(mbm.get_value())
            WU.set_value(mWU.get_value())
            bU.set_value(mbU.get_value())

        return net


    def visualize_weight_matrices(self, layer):
        """
        Displays the weight matrices of a specified layer and the uncertainties associated with them as images.
        """

        if isinstance(layer, int) and 0 <= layer < self.net.n_layers:

            fig, axs = plt.subplots(2, 2)

            im00 = axs[0, 0].imshow(self.net.mWs[layer].get_value(), cmap='gray', interpolation='none')
            im01 = axs[0, 1].imshow(np.exp(self.net.sWs[layer].get_value()), cmap='gray', interpolation='none')
            im10 = axs[1, 0].imshow(self.net.mbs[layer].get_value()[np.newaxis, :], cmap='gray', interpolation='none')
            im11 = axs[1, 1].imshow(np.exp(self.net.sbs[layer].get_value()[np.newaxis, :]), cmap='gray', interpolation='none')

            plt.colorbar(im00, ax=axs[0, 0])
            plt.colorbar(im01, ax=axs[0, 1])
            plt.colorbar(im10, ax=axs[1, 0])
            plt.colorbar(im11, ax=axs[1, 1])

            axs[0, 0].set_title('weight means')
            axs[0, 1].set_title('weight stds')
            axs[1, 0].set_title('bias means')
            axs[1, 1].set_title('bias stds')

        elif layer == 'a':

            fig, axs = plt.subplots(2, 2)

            im00 = axs[0, 0].imshow(self.mWa.get_value(), cmap='gray', interpolation='none')
            im01 = axs[0, 1].imshow(np.exp(self.sWa.get_value()), cmap='gray', interpolation='none')
            im10 = axs[1, 0].imshow(self.mba.get_value()[np.newaxis, :], cmap='gray', interpolation='none')
            im11 = axs[1, 1].imshow(np.exp(self.sba.get_value()[np.newaxis, :]), cmap='gray', interpolation='none')

            plt.colorbar(im00, ax=axs[0, 0])
            plt.colorbar(im01, ax=axs[0, 1])
            plt.colorbar(im10, ax=axs[1, 0])
            plt.colorbar(im11, ax=axs[1, 1])

            axs[0, 0].set_title('weight means')
            axs[0, 1].set_title('weight stds')
            axs[1, 0].set_title('bias means')
            axs[1, 1].set_title('bias stds')

        elif layer == 'm':

            for i in xrange(self.n_components):

                fig, axs = plt.subplots(2, 2)

                im00 = axs[0, 0].imshow(self.mWms[i].get_value(), cmap='gray', interpolation='none')
                im01 = axs[0, 1].imshow(np.exp(self.sWms[i].get_value()), cmap='gray', interpolation='none')
                im10 = axs[1, 0].imshow(self.mbms[i].get_value()[np.newaxis, :], cmap='gray', interpolation='none')
                im11 = axs[1, 1].imshow(np.exp(self.sbms[i].get_value()[np.newaxis, :]), cmap='gray', interpolation='none')

                plt.colorbar(im00, ax=axs[0, 0])
                plt.colorbar(im01, ax=axs[0, 1])
                plt.colorbar(im10, ax=axs[1, 0])
                plt.colorbar(im11, ax=axs[1, 1])

                axs[0, 0].set_title('weight means')
                axs[0, 1].set_title('weight stds')
                axs[1, 0].set_title('bias means')
                axs[1, 1].set_title('bias stds')

        elif layer == 'U':

            for i in xrange(self.n_components):

                fig, axs = plt.subplots(2, 2)

                im00 = axs[0, 0].imshow(self.mWUs[i].get_value(), cmap='gray', interpolation='none')
                im01 = axs[0, 1].imshow(np.exp(self.sWUs[i].get_value()), cmap='gray', interpolation='none')
                im10 = axs[1, 0].imshow(self.mbUs[i].get_value()[np.newaxis, :], cmap='gray', interpolation='none')
                im11 = axs[1, 1].imshow(np.exp(self.sbUs[i].get_value()[np.newaxis, :]), cmap='gray', interpolation='none')

                plt.colorbar(im00, ax=axs[0, 0])
                plt.colorbar(im01, ax=axs[0, 1])
                plt.colorbar(im10, ax=axs[1, 0])
                plt.colorbar(im11, ax=axs[1, 1])

                axs[0, 0].set_title('weight means')
                axs[0, 1].set_title('weight stds')
                axs[1, 0].set_title('bias means')
                axs[1, 1].set_title('bias stds')

        else:
            raise ValueError('Layer {} doesn\'t exist.'.format(layer))

        plt.show(block=False)


def replicate_gaussian_mdn(net, n_rep):
    """Takes an mdn with one component, and returns an mdn with that component replicated n_rep times."""

    mog_net = None

    if isinstance(net, MDN):

        assert net.n_components == 1

        mog_net = MDN(n_inputs=net.n_inputs, n_hiddens=net.net.n_units[1:], act_fun=net.act_fun, n_outputs=net.n_outputs, n_components=n_rep)

        for mog_p, p in izip(mog_net.net.parms, net.net.parms):
            mog_p.set_value(p.get_value())

        mog_net.Wa.set_value(np.zeros_like(mog_net.Wa.get_value()))
        mog_net.ba.set_value(np.zeros_like(mog_net.ba.get_value()))

        for Wm, bm, WU, bU in izip(mog_net.Wms, mog_net.bms, mog_net.WUs, mog_net.bUs, mog_net.Vs):
            Wm.set_value(net.Wms[0].get_value())
            bm.set_value(net.bms[0].get_value())
            WU.set_value(net.WUs[0].get_value())
            bU.set_value(net.bUs[0].get_value())
            V.set_value(net.Vs[0].get_value())

        for bm in mog_net.bms:
            bm.set_value(bm.get_value() + 1.0e-6 * rng.randn(*bm.get_value().shape).astype(dtype))

    elif isinstance(net, MDN_SVI):

        assert net.n_components == 1

        mog_net = MDN_SVI(n_inputs=net.n_inputs, n_hiddens=net.net.n_units[1:], act_fun=net.act_fun, n_outputs=net.n_outputs, n_components=n_rep)

        for mog_p, p in izip(mog_net.net.parms, net.net.parms):
            mog_p.set_value(p.get_value())

        mog_net.mWa.set_value(np.zeros_like(mog_net.mWa.get_value()))
        mog_net.mba.set_value(np.zeros_like(mog_net.mba.get_value()))
        mog_net.sWa.set_value(-5.0 * np.ones_like(mog_net.sWa.get_value()))
        mog_net.sba.set_value(-5.0 * np.ones_like(mog_net.sba.get_value()))

        for mWm, mbm, mWU, mbU, sWm, sbm, sWU, sbU, V in izip(mog_net.mWms, mog_net.mbms, mog_net.mWUs, mog_net.mbUs, mog_net.sWms, mog_net.sbms, mog_net.sWUs, mog_net.sbUs, mog_net.Vs):
            mWm.set_value(net.mWms[0].get_value())
            mbm.set_value(net.mbms[0].get_value())
            mWU.set_value(net.mWUs[0].get_value())
            mbU.set_value(net.mbUs[0].get_value())
            sWm.set_value(net.sWms[0].get_value())
            sbm.set_value(net.sbms[0].get_value())
            sWU.set_value(net.sWUs[0].get_value())
            sbU.set_value(net.sbUs[0].get_value())
            V.set_value(net.Vs[0].get_value())

        for mbm in mog_net.mbms:
            mbm.set_value(mbm.get_value() + 1.0e-6 * rng.randn(*mbm.get_value().shape).astype(dtype))

    else:
        ValueError('Net must be either an mdn or an svi mdn.')

    return mog_net
