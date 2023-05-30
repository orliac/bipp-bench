import argparse
import casacore.tables as ct
import astropy.table as tb
import astropy.time as time
import numpy as np

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ms", help="Name of measurement set", type=str)
    p.add_argument("--data_col", help="Column where data lives. "
                   "Only used to get shape of data at this stage",
                   default='DATA', type=str)
    p.add_argument("--time_start_idx", default=0, help="Index of first epoch to process.", type=int)
    p.add_argument("--time_end_idx",   default=1, help="Index of last epoch to process.", type=int)
    p.add_argument("--time_slice",     default=1, help="Time index slicing", type=int)
    p.add_argument("--channel_id", default=0, help="Channel ID to consider", type=int)
    args = p.parse_args()
    
    assert args.time_start_idx < args.time_end_idx

    return args


def visibilities(msf, channel_id, time_id, column):
        """
        Extract visibility matrices.

        Parameters
        ----------
        channel_id : array-like(int) or slice
            Several CHANNEL_IDs from :py:attr:`~pypeline.phased_array.util.measurement_set.MeasurementSet.channels`.
        time_id : int or slice
            Several TIME_IDs from :py:attr:`~pypeline.phased_array.util.measurement_set.MeasurementSet.time`.
        column : str
            Column name from MAIN table where visibility data resides.

            (This is required since several visibility-holding columns can co-exist.)

        Returns
        -------
        TBC
        """

        query = f"select * from {msf}"
        table = ct.taql(query)
        t = time.Time(np.unique(table.calc("MJD(TIME)")), format="mjd", scale="utc")
        t_id = range(len(t))
        _time = tb.QTable(dict(TIME_ID=t_id, TIME=t))
        print("-D- _time =", _time)


        if column not in ct.taql(f"select * from {msf}").colnames():
            raise ValueError(f"column={column} does not exist in {msf}::MAIN.")

        N_time = len(_time)
        time_start, time_stop, time_step = time_id.indices(N_time)
        print(time_start, time_stop, time_step)

        # Only a subset of the MAIN table's columns are needed to extract visibility information.
        # As such, it makes sense to construct a TaQL query that only extracts the columns of
        # interest as shown below:
        #    select ANTENNA1, ANTENNA2, MJD(TIME) as TIME, {column}, FLAG from {self._msf} where TIME in
        #    (select unique TIME from {self._msf} limit {time_start}:{time_stop}:{time_step})
        # Unfortunately this query consumes a lot of memory due to the column selection process.
        # Therefore, we will instead ask for all columns and only access those of interest.
        query = (
            f"select * from {msf} where TIME in "
            f"(select unique TIME from {msf} limit {time_start}:{time_stop}:{time_step})"
        )
        table = ct.taql(query)
        
        for sub_table in table.iter("TIME", sort=True):
            print(sub_table)
            print(sub_table.CDATETIME(TIME))
            t = time.Time(sub_table.calc("MJD(TIME)")[0], format="mjd", scale="utc")
            #t2 = time.Time(sub_table.TIME)")[0], format="datetime", scale="utc")
            print(t)
            

if __name__ == "__main__":
    args = get_args()
    print(args)

    time_id = slice(args.time_start_idx, args.time_end_idx, args.time_slice)
    mjds = visibilities(args.ms, args.channel_id, time_id, args.data_col)
