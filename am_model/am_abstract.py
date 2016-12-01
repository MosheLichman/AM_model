"""
Author: Moshe Lichman
"""
from __future__ import division
import abc
import numpy as np

from scipy.sparse import lil_matrix

from commons import helpers
from commons import log_utils as log
from commons import objectives
from commons import time_measure as tm


class AMAbstract(object):
    def __init__(self, U, I, num_comp=4, num_proc=1, user_smooth=None, item_smooth=None):
        self._U, self._I, self._num_comp, self._num_proc = U, I, num_comp, num_proc
        self._user_smooth, self._item_smooth = user_smooth, item_smooth

        log.info('Initializing model parameters')

        self._prev_mix_counts = np.zeros([U, num_comp])
        self._curr_mix_counts = np.zeros([U, num_comp])

        self._model_params = {'g_0': 10 / I, 'm_0': 1 / num_comp,
                              'sum_b': 1, 'eta': 1, 'mix': np.zeros([U, num_comp])}

        self._eps = 1E-12
        self._ga_step = 0.1

    def get_params(self):
        return self._model_params

    def training(self, train_data, val_data):
        """
        Learning the mixture weights (and the hyper parameters).

         INPUT:
        -------
            1. train_data:      <(U, L) lil_mat>    user observation data - used to learn the components
            2. val_data:        <(N, 2) ndarray>    validation events. Each row is a point [user_id, loc_id].
            3. return_mix:      <boolean>           returning the model mixture weights (default: False)

         OUTPUT:
        --------
            1. (optional) mix       <(U, k) ndarray>        mixture weights for each user
        """
        log.info('Running EM to optimize mixture weights.')
        g_mpe, mle, f_mle, l_mle = self._learn_components(train_data, np.unique(val_data[:, 0]))

        lamb = self._learn_lambda()

        self._em_opt(val_data, mle, g_mpe, l_mle, f_mle, lamb)

        for i in range(5):
            self._opt_hyper_params(f_mle, g_mpe, l_mle, lamb, mle, val_data)

        self._update_mix_weights(self._curr_mix_counts)

        self._prev_mix_counts += self._curr_mix_counts

        return self._model_params

    def evaluation(self, train_data, test_data, params=None, objective='logP', return_all=False):
        if params is None:
            params = self._model_params
        else:
            self._model_params = params

        g_mpe, mle, f_mle, l_mle = self._learn_components(train_data, np.unique(test_data[:, 0]))
        results = []

        uids = np.unique(test_data[:, 0]).astype(int)
        batch_size = int(np.ceil(uids.shape[0] / self._num_proc))

        helpers.quque_on_uids(self._num_proc, uids, batch_size, self._mp_user_test,
                              (objective, g_mpe, mle, f_mle, l_mle, test_data, params['lambda'], params['eta'],
                               params['sum_b'], True, return_all), results.extend)

        return results

    def _get_mix_counts_for_user(self, uid):
        return self._prev_mix_counts[uid, :], self._curr_mix_counts[uid, :]

    def _get_mix_probs_for_user(self, uid):
        return self._model_params['mix'][uid, :]

    def _opt_hyper_params(self, f_mle, g_mpe, l_mle, lamb, mle, val_data):
        log.info('Optimizing hyper parameters.')
        hyper_point = tm.get_point('hyper_learn')
        f_sum_b = []

        uids = np.unique(val_data[:, 0]).astype(int)
        batch_size = int(np.ceil(uids.shape[0] / self._num_proc))

        helpers.quque_on_uids(self._num_proc, uids, batch_size, self._mp_user_test,
                              ('logP', g_mpe, mle, f_mle, l_mle, val_data, lamb, self._model_params['eta'],
                               self._model_params['sum_b']), f_sum_b.extend)

        f_sum_b_eps = []
        helpers.quque_on_uids(self._num_proc, uids, batch_size, self._mp_user_test,
                              ('logP', g_mpe, mle, f_mle, l_mle, val_data, lamb, self._model_params['eta'],
                               self._model_params['sum_b'] * np.exp(self._eps)), f_sum_b_eps.extend)

        f_sum_b = np.array(f_sum_b)
        f_sum_b = np.mean(f_sum_b[:, 1])
        f_sum_b_eps = np.array(f_sum_b_eps)
        f_sum_b_eps = np.mean(f_sum_b_eps[:, 1])
        grad = (f_sum_b_eps - f_sum_b) / self._eps
        new_sum_b = self._model_params['sum_b'] * np.exp(self._ga_step * grad)
        log.debug('Gradient ascent on sum_dirch: %.3f -> %.3f' % (self._model_params['sum_b'], new_sum_b))
        self._model_params['sum_b'] = new_sum_b

        # Gradiant ascent step on sum_mg
        f_eta = []
        helpers.quque_on_uids(self._num_proc, uids, batch_size, self._mp_user_test,
                              ('logP', g_mpe, mle, f_mle, l_mle, val_data, lamb, self._model_params['eta'],
                               self._model_params['sum_b']), f_eta.extend)

        f_eta_eps = []
        helpers.quque_on_uids(self._num_proc, uids, batch_size, self._mp_user_test,
                              ('logP', g_mpe, mle, f_mle, l_mle, val_data, lamb,
                               self._model_params['eta'] * np.exp(self._eps), self._model_params['sum_b']),
                              f_eta_eps.extend)

        f_eta = np.array(f_eta)
        f_eta = np.mean(f_eta[:, 1])
        f_eta_eps = np.array(f_eta_eps)
        f_eta_eps = np.mean(f_eta_eps[:, 1])
        grad = (f_eta_eps - f_eta) / self._eps
        new_eta = self._model_params['eta'] * np.exp(self._ga_step * grad)  # It's gradient ascent
        log.debug('Gradient ascent on sum_mg: %.3f -> %.3f' % (self._model_params['eta'], new_eta))
        self._model_params['eta'] = new_eta

        self._model_params['lambda'] = self._learn_lambda()
        hyper_point.collect()

    def _learn_components(self, train_data, relevant_uids):
        learn_point = tm.get_point('learn')
        train_global = np.array(train_data.sum(axis=0))[0]

        g_mpe = train_global + self._model_params['g_0']
        g_mpe /= np.sum(g_mpe)

        rel_data = train_data[relevant_uids, :].copy()

        mle_point = tm.get_point('learn_mle')
        nnz = np.where(rel_data.sum(axis=1) > 0)[0]
        tmp = rel_data[nnz] / rel_data[nnz].sum(axis=1)

        mle = lil_matrix((self._U, self._I))
        mle[relevant_uids] = tmp
        mle = mle.tocsr()
        mle_point.collect()

        if self._num_comp == 2:
            learn_point.collect()
            return g_mpe, mle, None, None

        lmle_point = tm.get_point('learn_lmle')
        tmp = lil_matrix(train_data[relevant_uids].tocsr().dot(self._item_smooth))
        nnz = np.where(tmp.sum(axis=1) > 0)[0]
        tmp[nnz] /= tmp[nnz].sum(axis=1)

        l_mle = lil_matrix((self._U, self._I))
        l_mle[relevant_uids] = tmp
        l_mle = l_mle.tocsr()
        lmle_point.collect()

        if self._num_comp == 3:
            learn_point.collect()
            return g_mpe, mle, None, l_mle

        fmle_point = tm.get_point('learm_fmle')

        tmp = lil_matrix(self._user_smooth[relevant_uids].dot(train_data.tocsc()))
        nnz = np.where(tmp.sum(axis=1) > 0)[0]
        tmp[nnz] /= tmp[nnz].sum(axis=1)

        f_mle = lil_matrix((self._U, self._I))
        f_mle[relevant_uids] = tmp
        f_mle = f_mle.tocsr()
        fmle_point.collect()

        learn_point.collect()

        return g_mpe, mle, f_mle, l_mle

    def _update_mix_weights(self, curr_mix_counts):
        # The m_0 has changed, and so did the lamb
        lamb = self._learn_lambda()

        mask = np.where(np.sum(curr_mix_counts, axis=1) > 0)[0]
        prior = self._prev_mix_counts[mask, :] + lamb * self._model_params['eta']
        prior /= np.reshape(np.sum(prior, axis=1), [prior.shape[0], 1])
        prior *= self._model_params['sum_b']
        self._model_params['mix'][mask, :] = curr_mix_counts[mask, :] + prior
        self._model_params['mix'][mask, :] /= \
            np.reshape(np.sum(self._model_params['mix'][mask, :], axis=1), [mask.shape[0], 1])

    def _learn_lambda(self):
        lamb = np.sum(self._prev_mix_counts, axis=0) + self._model_params['m_0']
        lamb /= np.sum(lamb)
        return lamb

    def _mp_user_test(self, queue, uids, args):
        """
        Process wise evaluation of the model. Can be either called with all the parameters that are used to compute the
        mixture weights - used in the hyper parameters learning where we want to do gradient ascent.
        Alternatively, it can be called using the learned mixture weights (saved by the instance). The latter is used
        in evaluation steps.

         INPUT:
        -------
            1. objective:       <string>            objective function to use for evaluation
            2. uids:            <(U_b, ) ndarray>   batch uids to test in the process
            3. train_data:      <(U, L) csr_mat>    user observation data - used to learn the components
            4. test_data:       <(N, 2) ndarray>    each row is a test event [user_id, loc_id]
            5. g_MPE:           <(L, ) ndarray>     learned global components
            6. lamb:            <(k, ) ndarray>     learned mixture weights population prior
            7. eta:             <float>             strength of the mixture weights population prior
            8. sum_b:           <float>             strength of the mixture weights individual prior
            9. use_mix:         <boolean>           use learned mixture weights

         OUTPUT:
        --------
           1. results:      <list>      each row is [user_id, avg score]
        """
        use_mix = False
        return_all = False

        if len(args) == 9:
            objective, g_MPE, MLE, f_MLE, l_MLE, test_data, lamb, eta, sum_b = args
        if len(args) == 10:
            objective, g_MPE, MLE, f_MLE, l_MLE, test_data, lamb, eta, sum_b, return_all = args
        if len(args) == 11:
            objective, g_MPE, MLE, f_MLE, l_MLE, test_data, lamb, eta, sum_b, use_mix, return_all = args
        try:
            results = []

            for i in range(len(uids)):
                user_eval_point = tm.get_point('user_eval')
                uid = uids[i]
                u_test = test_data[np.where(test_data[:, 0] == uid)[0], 1].astype(int)

                # First computing the users clusters
                u_mle = MLE[uid, :]

                if not use_mix:
                    prev_counts, curr_counts = self._get_mix_counts_for_user(uid)
                    prior = prev_counts + lamb * eta
                    prior /= np.sum(prior)
                    prior *= sum_b

                    m_counts = curr_counts
                    c_prob = (m_counts + prior) / np.sum(m_counts + prior)
                else:
                    c_prob = self._get_mix_probs_for_user(uid)

                MLE_probs = c_prob[0] * np.array(u_mle[0, :].toarray())[0]
                g_probs = c_prob[1] * g_MPE

                if self._num_comp == 2:
                    user_mult = MLE_probs + g_probs
                elif self._num_comp == 3:
                    ul_mle = l_MLE[uid, :]
                    l_probs = c_prob[2] * np.array(ul_mle[0, :].toarray())[0]
                    user_mult = MLE_probs + g_probs + l_probs
                else:
                    ul_mle = l_MLE[uid, :]
                    uf_mle = f_MLE[uid, :]
                    l_probs = c_prob[2] * np.array(ul_mle[0, :].toarray())[0]
                    f_probs = c_prob[3] * np.array(uf_mle[0, :].toarray())[0]
                    user_mult = MLE_probs + g_probs + l_probs + f_probs

                results.append([uid, objectives.obj_func[objective](user_mult, u_test, return_all)])
                user_eval_point.collect()

            queue.put(results)
        except Exception as e:
            print 'TEST: Error for user %d' % uid
            print e

    @abc.abstractmethod
    def _em_opt(self, val_data, mle, g_mpe, l_mle, f_mle, lamb):
        raise NotImplementedError('Abstract class.')
