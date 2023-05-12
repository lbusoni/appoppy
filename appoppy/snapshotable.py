import abc
import astropy.io.fits as pyfits
from six import with_metaclass
import numpy as np


class SnapshotPrefix(object):
    PETALOMETER = "PET"
    PATH1 = "PATH1"
    PATH2 = "PATH2"
    PHASE_SHIFT_INTERFEROMETER = "PSI"
    PASSATA_RESIDUAL = "PASSATA"
    LOW_WIND_EFFECT = "LWE"


class Snapshotable(with_metaclass(abc.ABCMeta, object)):

    @abc.abstractmethod
    def get_snapshot(self, prefix):
        assert False

    @staticmethod
    def _is_suitable_for_fits_header(value):
        return value is not None

    @staticmethod
    def _replace_infinity(value):
        if value == np.inf:
            return 1e300
        elif value == -np.inf:
            return -1e300
        else:
            return value

    @staticmethod
    def _truncate_string_if_any(key, value):
        maxValueLenInChars = 67 - len(key)
        if len(value) > maxValueLenInChars:
            return value[0:maxValueLenInChars]
        return value

    @staticmethod
    def _update_header(hdr, key, value):
        MAX_KEY_LEN_CHARS = 59
        assert len(key) <= MAX_KEY_LEN_CHARS
        if isinstance(value, str):
            value = Snapshotable._truncate_string_if_any(key, value)
        try:
            hdr.update({'hierarch ' + key: value})
        except ValueError:
            hdr.update({'hierarch ' + key: str(value)})

    @staticmethod
    def as_fits_header(snapshotDictionary):
        hdr = pyfits.Header()
        for k in sorted(snapshotDictionary.keys()):
            value = snapshotDictionary[k]
            if Snapshotable._is_suitable_for_fits_header(value):
                Snapshotable._update_header(hdr, k, value)
        return hdr

    @staticmethod
    def prepend(prefix, snapshotDict):
        assert len(prefix) > 0, "Prefix length must be greater than zero"
        for each in list(snapshotDict.keys()):
            value = snapshotDict[each]
            del snapshotDict[each]
            newKey = prefix + "." + each
            snapshotDict[newKey] = value
        return snapshotDict

    @staticmethod
    def from_fits_header(hdr):
        snapshot = {}
        for each in hdr:
            snapshot[each] = hdr[each]
        return snapshot

    @staticmethod
    def remove_entries_with_value_none(snapshotDict):
        for each in list(snapshotDict.keys()):
            if snapshotDict[each] is None:
                del snapshotDict[each]
