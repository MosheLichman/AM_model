"""
Author: Moshe Lichman
"""
from __future__ import division
from multiprocessing import Process, Queue


def quque_on_uids(num_proc, uids, batch_size, target, args, collect_func):
    queue = Queue()
    if num_proc > 1:
        proc_pool = []
        for i in range(num_proc):
            p_uids = uids[i * batch_size:(i + 1) * batch_size]
            if len(p_uids) == 0:
                break

            # Adding the p_uids and queue to args
            proc = Process(target=target, args=(queue, p_uids, args))
            proc_pool.append(proc)

        [proc.start() for proc in proc_pool]
        for _ in range(len(proc_pool)):
            resp = queue.get()
            collect_func(resp)

        [proc.join() for proc in proc_pool]
    else:
        target(queue, uids, args)
        resp = queue.get()
        collect_func(resp)
