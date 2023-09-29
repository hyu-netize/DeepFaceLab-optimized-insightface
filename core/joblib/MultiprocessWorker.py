# 2023 - Created by Dmitry Kalashnik.

import multiprocessing

from core.interact import interact as io

PROGRESS_BAR_COUNTER_SIZE = 7


class MultiprocessWorker(object):

    def __init__(self, processes=multiprocessing.cpu_count(), preserve_pool=False):
        self.pool = None
        self.processes_count = processes
        self.preserve_pool = preserve_pool
        self.progress_bar_enabled = False
        self.progress_bar_message = None
        self.leave_progress_bar = True
        self.progress_bar_counter = 0

    def try_show_progress_bar(self, input_size):
        if self.progress_bar_enabled:
            io.progress_bar(self.progress_bar_message, input_size)

    def try_increase_progress_bar(self):
        if not self.progress_bar_enabled:
            return

        self.progress_bar_counter += 1
        if self.progress_bar_counter == PROGRESS_BAR_COUNTER_SIZE:
            self.progress_bar_counter = 0
            io.progress_bar_inc(PROGRESS_BAR_COUNTER_SIZE)

    def close(self):
        if self.progress_bar_enabled:
            self.progress_bar_enabled = False
            self.progress_bar_message = None
            io.progress_bar_close()
        if not self.preserve_pool and self.pool is not None:
            self.pool.close()

    def close_pool(self):
        if self.pool is not None:
            self.pool.close()
            self.pool.join()

    def schedule_tasks(self, func, iterable_input, args):
        self.try_show_progress_bar(len(iterable_input))
        if self.pool is None:
            self.pool = multiprocessing.Pool(self.processes_count)

        futures = []
        for entry in iterable_input:
            futures.append(self.pool.apply_async(func, args=(entry, *args)))
        return futures

    def run(self, func, iterable_input, *args):
        futures = self.schedule_tasks(func, iterable_input, args)
        result = []
        for incomplete_future in futures:
            result.append(incomplete_future.get())
            self.try_increase_progress_bar()

        self.close()
        return result

    def run_sorting(self, func, iterable_input, *args):
        futures = self.schedule_tasks(func, iterable_input, args)
        result_processed = []
        result_error = []

        for incomplete_future in futures:
            completed_result = incomplete_future.get()
            success = completed_result[0]
            if success:
                result_processed.append(completed_result[1])
            else:
                result_error.append(completed_result[1])
            self.try_increase_progress_bar()

        self.close()
        return result_processed, result_error

    def set_progress_bar(self, message, leave=True):
        self.progress_bar_message = message
        self.leave_progress_bar = leave
        self.progress_bar_enabled = True
