import numpy as np

import matplotlib.pyplot as plt
from configs.configs import logger, DEBUG
from configs.modelConfigs import CHORUS_DURATION_SINGLE, CHORUS_DURATION, TUNE_SCOPE
from utility.common import (
    mergeIntervals,
    intervalIntersection,
    singleChorusSection,
    multiChorusSections,
    filterIntvs,
)


def maxOverlap(mirexFmt, chorusDur=CHORUS_DURATION_SINGLE, centering=False):
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


def arousalPoint(time, times, pitches, window, show=DEBUG):
    def arousalScore(t):
        beforePitches = pitches[(times >= t - TUNE_SCOPE / 2) & (times <= t)]
        afterPitches = pitches[(times >= t) & (times <= t + TUNE_SCOPE / 2)]
        score = np.sum(afterPitches) - np.sum(beforePitches)
        return score / len(beforePitches)

    mask = (times >= time - window / 2) & (times <= time + window / 2)
    scores = [arousalScore(t) for t in times[mask]]
    point = times[mask][np.argmax(scores)]
    if show:
        logger.debug(
            f"point={point} times={times[mask][0]}~{times[mask][-1]} window={window}"
        )
        plt.plot(times[mask], pitches[mask], label="pitch")
        plt.plot(times[mask], scores, label="score")
        plt.scatter(point, np.max(scores))
        plt.xlabel("time/s")
        plt.ylabel("freq/Hz")
        plt.legend()
        plt.show()
    return point


def tuneIntervals(mirexFmt, mels_f, chorusDur, window):
    mirexFmt = mergeIntervals(mirexFmt)
    dur = mirexFmt[0][-1][1]
    intvs = filterIntvs(mirexFmt, fun="chorus")
    tuneIntvs = []
    times, pitches = mels_f
    for intv in intvs:
        begin = arousalPoint(intv[0], times, pitches, window)
        end = arousalPoint(intv[1], times, -pitches, window)
        end = min(dur, max(end, begin + chorusDur))
        tuneIntvs.append((begin, end))
    return multiChorusSections(tuneIntvs, dur)
