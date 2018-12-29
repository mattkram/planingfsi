"""A simple stopwatch for measuring walltime."""
import datetime


class Stopwatch(object):
    """A simple class representing a stopwatch to measure wall clock time.

    Parameters
    ----------
    start : bool, optional
        If True, the Stopwatch will be started on instantiation.

    Attributes
    ----------
    start_time : datetime
        The time the Stopwatch was started.
    stop_time : datetime
        The time the Stopwatch was stopped.

    """

    def __init__(self, start=False):
        self.start_time = None
        self.stop_time = None
        if start:
            self.start()

    def start(self):
        """Start the clock."""
        self.start_time = datetime.datetime.now()

    def stop(self):
        """Stop the clock."""
        self.stop_time = datetime.datetime.now()

    def get_elapsed_time(self):
        """Return the Stopwatch elapsed time.

        If the stop time is set, return the difference between stop and start time. Otherwise,
        use the current time as the stop time.

        Returns
        -------
        datetime.datetime
            The elapsed time.

        """
        if self.stop_time is None:
            return datetime.datetime.now() - self.start_time
        return self.stop_time - self.start_time

    def get_elapsed_seconds(self):
        """Return the elapsed time in seconds."""
        return self.get_elapsed_time().total_seconds()

    def print_elapsed_time(self, stop=False, print_func=print):
        """Print the elapsed time to the screen.

        Parameters
        ----------
        stop : bool, optional
            If True, the Stopwatch will also be stopped when printing the elapsed time.
        print_func : function, optional
            A logger function to handle logging. Defaults to print function.

        """
        if stop:
            self.stop()
        minutes, seconds = divmod(self.get_elapsed_seconds(), 60)
        hours, minutes = divmod(minutes, 60)

        print_func(
            "\nExecution completed in %d hours, %d minutes, %d seconds",
            hours,
            minutes,
            seconds,
        )
