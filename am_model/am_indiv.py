"""
Author: Moshe Lichman
"""
from __future__ import division
import numpy as np

from am_model import am_abstract

from commons import helpers
from commons import log_utils as log
from commons import time_measure as tm


class AMIndiv(am_abstract.AMAbstract):

    def _em_opt(self, val_data, mle, g_mpe, l_mle, f_mle, lamb):
        em_point = tm.get_point('em')
        uids = np.unique(val_data[:, 0]).astype(int)
        log.debug('Validation data stats: [%d events] [%d users].' % (val_data.shape[0], len(uids)))
        batch_size = int(np.ceil(uids.shape[0] / self._num_proc))
        mixing_weights_and_uids = []

        helpers.quque_on_uids(self._num_proc, uids, batch_size, self._mp_user_em,
                      (g_mpe, mle, f_mle, l_mle, val_data, lamb, self._model_params['eta'],
                       self._model_params['sum_b']), mixing_weights_and_uids.append)

        em_point.collect()

        curr_mix_counts = np.zeros([self._U, self._num_comp])
        for uids, mixing_counts in mixing_weights_and_uids:
            curr_mix_counts[uids, :] = mixing_counts

        self._curr_mix_counts = curr_mix_counts

    def _mp_user_em(self, queue, uids, args):
        """
        Using EM algorithm to optimize the mixture weights.

         INPUT:
        -------
            1. uids:        <(U_b, ) ndarray>   batch uids to optimize on
            2. train_data:  <(U, L) csr_mat>    user observation data - used to learn the components
            2. opt_data:    <(D_va, 2 ndarray>  each row is a validation data point [uid, lid]
            3. g_MPE:       <(L, ) ndarray>     global probabilities.
            4. lamb:        <(k, ) ndarray>     global mixture weight prior(can't use lambda)
            5. eta:         <float>             strength of lamb
            6. sum_b:       <float>             strength of individual mixture weights prior

         OUTPUT:
        --------
            1. uids:         <(U_b, ) ndarray      batch uids to optimize on
            2. all_m_counts: <(U_b, K) ndarray>    mixing counts for each of the users.
        """
        g_MPE, MLE, f_MLE, l_MLE, val_data, lamb, eta, sum_b = args
        try:
            all_m_counts = np.zeros([len(uids), self._num_comp])
            for i in range(len(uids)):
                em_point = tm.get_point('user_em')
                uid = uids[i]
                u_val = val_data[np.where(val_data[:, 0] == uid)[0], 1].astype(int)

                u_mle = MLE[uid, :]

                prior = self._prev_mix_counts[uid, :] + lamb * eta
                prior /= np.sum(prior)
                prior *= sum_b

                # Initializing the component probability to be the prior
                c_prob = prior / np.sum(prior)

                log_like = -np.inf

                for em_iter in range(20):
                    # E-Step -- the Bayes probabilities
                    MLE_probs = c_prob[0] * np.array(u_mle[0, u_val].toarray())[0]
                    g_probs = c_prob[1] * g_MPE[u_val]

                    if self._num_comp == 2:
                        probs = [MLE_probs, g_probs]
                    else:
                        uf_mle = f_MLE[uid, :]
                        ul_mle = l_MLE[uid, :]

                        l_probs = c_prob[2] * np.array(ul_mle[0, u_val].toarray())[0]
                        f_probs = c_prob[3] * np.array(uf_mle[0, u_val].toarray())[0]

                        probs = [MLE_probs, g_probs, l_probs, f_probs]

                    probs = np.array(probs)
                    probs = probs.T

                    new_ll = np.sum(np.log(np.sum(probs, axis=1)))
                    if np.abs(new_ll - log_like) < 0.001:
                        break
                    else:
                        log_like = new_ll

                    probs /= np.reshape(np.sum(probs, axis=1), [u_val.shape[0], 1])

                    # M-Step -- only on the mixing weights, the components are fixed
                    m_counts = np.sum(probs, axis=0)
                    c_prob = (m_counts + prior) / np.sum(m_counts + prior)

                all_m_counts[i, :] = m_counts
                em_point.collect()

            queue.put([uids, all_m_counts])
        except Exception as e:
            log.critical('Failed on EM: user %d - %s' % (uid, e.message))
