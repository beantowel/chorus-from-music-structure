import numpy as np

import matplotlib.pyplot as plt
from configs.configs import logger, DEBUG
from utility.common import mergeIntervals, intervalIntersection


def maxOverlap(mirexFmt, chorusDur=30.0):
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

    begin = intervals[idx][0]
    end = min(dur, begin + chorusDur)
    intervals = np.array([(0, begin), (begin, end), (end, dur),])
    labels = np.array(["others", "chorus", "others",], dtype="U16")
    return (intervals, labels)


def arousalScore(t, times, pitches, win=5):
    beforePitches = pitches[(times >= t - win) & (times <= t)]
    afterPitches = pitches[(times >= t) & (times <= t + win)]
    return np.sum(afterPitches) - np.sum(beforePitches)


def maxArousal(mirexFmt, mels_f, chorusDur=30.0, window=10.0, show=DEBUG):

    intervals, labels = maxOverlap(mirexFmt, chorusDur=30.0)
    chorusIdx = 1
    # intervals, labels = mergeIntervals(mirexFmt)
    # chorusIdx = np.nonzero(np.char.startswith(labels, "chorus"))[0][0]

    dur = intervals[-1][1]
    times, pitches = mels_f

    begin = intervals[chorusIdx][0]
    mask = (times >= begin - window) & (times <= begin + window / 2)
    scores = [arousalScore(t, times, pitches) for t in times[mask]]
    if show:
        plt.plot(pitches[mask])
        plt.show()

    begin = times[mask][np.argmax(scores)]
    end = min(dur, max(intervals[chorusIdx][1], begin + chorusDur))
    logger.debug(
        f"begin={begin} end={end} shift={begin-intervals[1][0]} intervals={intervals}"
    )
    intervals = np.array([(0, begin), (begin, end), (end, dur),])
    labels = np.array(["others", "chorus", "others",], dtype="U16")
    return (intervals, labels)
