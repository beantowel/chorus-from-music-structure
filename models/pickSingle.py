import numpy as np

import matplotlib.pyplot as plt
from configs.configs import logger, DEBUG
from utility.common import (
    mergeIntervals,
    intervalIntersection,
    singleChorusSection,
    multiChorusSections,
    filterIntvs,
)


def maxOverlap(mirexFmt, chorusDur=30.0, centering=False):
    intervals, labels = mergeIntervals(mirexFmt)
    chorusIndices = np.nonzero(np.char.startswith(labels, "chorus"))[0]
    dur = intervals[-1][1]

    chorusIntsec = (
        []
    )  # select (begin, begin + 30s) with maximal overlap with detected chorus sections
    for idx in chorusIndices:
        begin = intervals[idx][0]
        end = min(dur, begin + chorusDur)
        intsec = np.sum(
            [intervalIntersection((begin, end), intervals[j]) for j in chorusIndices]
        )
        chorusIntsec.append(intsec)
    selectIndex = np.argmax(chorusIntsec)
    idx = chorusIndices[selectIndex]

    if not centering:
        begin = intervals[idx][0]
        end = min(dur, begin + chorusDur)
    else:
        center = np.mean(intervals[idx])
        begin = center - chorusDur / 2
        end = center + chorusDur / 2
    return singleChorusSection(begin, end, dur)


def arousalPoint(time, times, pitches, window, show=False):
    def arousalScore(t):
        beforePitches = pitches[(times >= t - window) & (times <= t)]
        afterPitches = pitches[(times >= t) & (times <= t + window)]
        return np.sum(afterPitches) - np.sum(beforePitches)

    mask = (times >= time - window / 2) & (times <= time + window / 2)
    scores = [arousalScore(t) for t in times[mask]]
    point = times[mask][np.argmax(scores)]
    if show:
        plt.plot(pitches[mask])
        plt.show()
    return point


def tuneIntervals(mirexFmt, mels_f, chorusDur=10.0, window=6.0):
    mirexFmt = mergeIntervals(mirexFmt)
    intvs = filterIntvs(mirexFmt, fun="chorus")
    dur = intvs[-1][1]
    tuneIntvs = []
    times, pitches = mels_f
    for intv in intvs:
        begin = arousalPoint(intv[0], times, pitches, window)
        end = arousalPoint(intv[1], times, -pitches, window)
        end = min(dur, max(end, begin + chorusDur))
        tuneIntvs.append((begin, end))
    return multiChorusSections(tuneIntvs, dur)
