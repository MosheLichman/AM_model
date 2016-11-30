"""
Author: Moshe Lichman
"""
from __future__ import division
import numpy as np

from am_model import am_abstract

from commons import time_measure as tm


class AMGlobal(am_abstract.AMAbstract):

    def _em_opt(self, val_data, mle, g_mpe, l_mle, f_mle, lamb):
        em_point = tm.get_point('em_global')
        prior = self._prev_mix_counts[0] + lamb * self._model_params['sum_b']
        prior /= np.sum(prior)
        prior *= self._model_params['sum_b']

        # Initializing the component probability to be the prior
        c_prob = prior / np.sum(prior)

        log_like = -np.inf

        for em_iter in range(20):
            # E-Step -- the Bayes probabilities
            MLE_probs = c_prob[0] * np.array(mle[val_data[:, 0].astype(int), val_data[:, 1].astype(int)])[0]
            g_probs = c_prob[1] * g_mpe[val_data[:, 1].astype(int)]

            if self._num_comp == 2:
                probs = [MLE_probs, g_probs]
            elif self._num_comp == 3:
                l_probs = c_prob[2] * np.array(l_mle[val_data[:, 0].astype(int), val_data[:, 1].astype(int)])[0]
                probs = [MLE_probs, g_probs, l_probs]
            else:
                l_probs = c_prob[2] * np.array(l_mle[val_data[:, 0].astype(int), val_data[:, 1].astype(int)])[0]
                f_probs = c_prob[3] * np.array(f_mle[val_data[:, 0].astype(int), val_data[:, 1].astype(int)])[0]

                probs = [MLE_probs, g_probs, l_probs, f_probs]

            probs = np.array(probs)
            probs = probs.T

            new_ll = np.sum(np.log(np.sum(probs, axis=1)))
            if np.abs(new_ll - log_like) < 0.001:
                break
            else:
                log_like = new_ll

            probs /= np.reshape(np.sum(probs, axis=1), [val_data.shape[0], 1])

            # M-Step -- only on the mixing weights, the components are fixed
            m_counts = np.sum(probs, axis=0)
            c_prob = (m_counts + prior) / np.sum(m_counts + prior)

        # By doing this I'm making sure that the update will happen for everyone and will be exactly the same
        self._curr_mix_counts[:] = m_counts
        em_point.collect()

