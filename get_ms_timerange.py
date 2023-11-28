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
    p.add_argument("--time_file", required=True, help="Path to file containing start and end datetimes to process in CASA format.")
    args = p.parse_args()
    
    assert args.time_start_idx < args.time_end_idx
    
    print(args)

    return args


def times(msf, time_id, column):
        """
        Extract datetimes

        Parameters
        ----------
        time_id : int or slice
            Several TIME_IDs from :py:attr:`~pypeline.phased_array.util.measurement_set.MeasurementSet.time`.
        column : str
            Column name from MAIN table where visibility data resides.

            (This is required since several visibility-holding columns can co-exist.)

        Returns
        -------
        Nothing. Write a two lines file with start and end datetimes in CASA format.
        """

        query = f"select * from {msf}"
        table = ct.taql(query)
        t = time.Time(np.unique(table.calc("MJD(TIME)")), format="mjd", scale="utc")
        t_id = range(len(t))
        _time = tb.QTable(dict(TIME_ID=t_id, TIME=t))
        #print("-D- _time =", _time)


        if column not in ct.taql(f"select * from {msf}").colnames():
            raise ValueError(f"column={column} does not exist in {msf}::MAIN.")

        N_time = len(_time)
        time_start, time_stop, time_step = time_id.indices(N_time)
        #print("-D time_start, time_stop, time_step =", time_start, time_stop, time_step)

        query = (
            f"select * from {msf} where TIME in "
            f"(select unique TIME from {msf} limit {time_start}:{time_stop}:{time_step})"
        )
        table = ct.taql(query)
        mjd_min = 1E10
        mjd_max = -1E10
        for sub_table in table.iter("TIME", sort=True):
            mjd = sub_table.calc("MJD(TIME)")[0]
            if mjd > mjd_max: mjd_max = mjd
            if mjd < mjd_min: mjd_min = mjd
        #print(f"-I- mjd = [{mjd_min:.7f}, {mjd_max:.7f}]")
        mjd_min -= 0.1 / 86400
        mjd_max += 0.1 / 86400
        t_min = time.Time(mjd_min, format="mjd", scale="utc")
        t_max = time.Time(mjd_max, format="mjd", scale="utc")
        t_min.format = 'isot'
        t_max.format = 'isot'
        #print(t_min, t_max)
        with open(args.time_file, "w") as f:
            f.write(isot_to_ms(t_min.value) + "\n")
            f.write(isot_to_ms(t_max.value) + "\n")


# Convert isot datetime string to format expected by CASA: YYYY/MM/DD/HH:MM:SS.FF
# https://casadocs.readthedocs.io/en/v6.3.0/notebooks/visibility_data_selection.html#The-timerange-Parameter
def isot_to_ms(t):
    t = t.replace('-','/',2)
    t = t.replace('T','/',1)
    #t = t[:-1]
    return t


if __name__ == "__main__":
    args = get_args()
    #print(args)

    time_id = slice(args.time_start_idx, args.time_end_idx, args.time_slice)
    times(args.ms, time_id, args.data_col)
