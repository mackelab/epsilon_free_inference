import numpy as np
import matplotlib.pyplot as plt
import mdn
import LossFunction as lf
import Trainer
import helper

def train_proposal_prior_with_bootstrapping(prior, x_obs, calc_dist, sim_likelihood,
                                            n_samples=200,n_bootstrap_iter=4,
                                            translate_priors=True,
                                            save=True, savefile=None):
    """Trains an svi mdn to return the posterior with boostrapping."""

    n_components = 1
    n_outputs = 1

    Vs = [0. * np.ones((n_outputs,n_outputs)) for i in xrange(n_components)]

    # create an mdn
    net = mdn.MDN_SVI(n_inputs=1, n_hiddens=[20], act_fun='tanh', n_outputs=n_outputs, 
                      n_components=n_components, Vs = Vs)
    regularizer = lf.regularizerSvi(net.mps, net.sps, 0.01)
    prior_proposal = None

    for iter in xrange(n_bootstrap_iter):

        # generate new data
        ms = np.empty(n_samples)
        xs = np.empty(n_samples)
        dist = np.empty(n_samples)

        for i in xrange(n_samples):

            ms[i] = prior['mu_a'] - 1.0
            while ms[i] < prior['mu_a'] or ms[i] > prior['mu_b']:
                ms[i] = prior['sim_prior']() if iter == 0 else prior_proposal.gen()[0]
            xs[i] = sim_likelihood(ms[i])
            dist[i] = calc_dist(xs[i], x_obs)

            print 'simulation {0}, distance = {1}'.format(i, dist[i])

        # plot distance histogram
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.hist(dist, bins=int(np.sqrt(n_samples)))
        ax.set_title('iteration = {0}'.format(iter + 1))
        ax.set_xlim([0.0, 12.0])
        plt.show(block=False)

        # train an mdn to give the posterior
        minibatch = 50
        maxiter = int(1000 * n_samples / minibatch)
        monitor_every = 100

        print('Vs pre set', [net.Vs[i].eval() for i in xrange(n_components)])
        net.set_sqrt_prior_precisions(Vs)
        print('Vs post set', [net.Vs[i].eval() for i in xrange(n_components)])


        trainer = Trainer.Trainer(
            model=net,
            trn_data=[xs[:, np.newaxis], ms[:, np.newaxis]],
            trn_loss=net.mlprob + regularizer / n_samples,
            trn_target=net.y
        )
        trainer.train(
            maxiter=maxiter,
            minibatch=minibatch,
            show_progress=True,
            monitor_every=monitor_every
        )

        # calculate the approximate posterior
        mdn_mog = net.get_mog(np.asarray([x_obs]), n_samples=None)
        mdn_mog.prune_negligible_components(1.0e-3)
        if translate_priors:
            if iter > 0:
                print('posterior precisions (post-fit)', [x.P for x in mdn_mog.xs])
                print('prior precision (post-fit)', prior_proposal.P)
            approx_posterior = mdn_mog if iter == 0 else mdn_mog / prior_proposal
        else:
            approx_posterior = mdn_mog if iter < (n_bootstrap_iter -1) else mdn_mog / prior_proposal
        prior_proposal = approx_posterior.project_to_gaussian()

        Vs = [np.linalg.inv(prior_proposal.C.T).reshape(n_outputs,n_outputs) for i in xrange(n_components)]
        #Vs = [0. * np.ones((n_outputs,n_outputs)) for i in xrange(n_components)]
        print('Vs post fit', Vs)



        # save the net and the approximate posterior
        if save:
            assert not savefile is None
            helper.save((net, approx_posterior, prior_proposal, dist), savefile.format(iter))

    return net

def train_mdn_with_proposal(loadfile, x_obs, sim_likelihood,
                            n_components=2,
                            n_samples=10000,
                            minibatch=100,
                            save=True,savefile=None):
    """Use the prior proposal learnt by bootstrapping to train an mdn."""

    # load prior proposal
    net, _, prior_proposal, _ = helper.load(loadfile)


    net = mdn.replicate_gaussian_mdn(net, n_components)

    # generate data
    ms = prior_proposal.gen(n_samples)
    xs = sim_likelihood(ms[:, 0])[:, np.newaxis]

    # train an mdn to give the posterior
    maxiter = int(20 * n_samples / minibatch)
    monitor_every = 1000
    regularizer = lf.regularizerSvi(net.mps, net.sps, 0.01)
    trainer = Trainer.Trainer(
        model=net,
        trn_data=[xs, ms],
        trn_loss=net.mlprob + regularizer / n_samples,
        trn_target=net.y
    )
    trainer.train(
        maxiter=maxiter,
        minibatch=minibatch,
        show_progress=True,
        monitor_every=monitor_every
    )

    # calculate the approximate posterior
    mdn_mog = net.get_mog(np.asarray([x_obs]))
    mdn_mog.prune_negligible_components(0.)
    approx_posterior = mdn_mog / prior_proposal

    # save the net
    if save:
        assert not savefile is None
        helper.save((net, approx_posterior, prior_proposal, mdn_mog), savefile)

    return net