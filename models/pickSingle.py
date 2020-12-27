import numpy as np
import librosa

import matplotlib.pyplot as plt
from configs.configs import logger, DEBUG
from configs.modelConfigs import (
    CHORUS_DURATION_SINGLE,
    CHORUS_DURATION,
    TUNE_SCOPE,
    CLF_TARGET_LABEL,
    MINIMUM_CHORUS_DUR,
)
from utility.common import (
    mergeIntervals,
    intervalIntersection,
    singleChorusSection,
    multiChorusSections,
    removeNumber,
    filterIntvs,
    mirexLines,
)


def maxOverlap(mirexFmt, chorusDur=CHORUS_DURATION_SINGLE, centering=False):
    intervals, labels = mergeIntervals(mirexFmt)
    chorusIndices = np.nonzero(np.char.startswith(labels, CLF_TARGET_LABEL))[0]
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


def arousalPoint(time, times, pitches, window, begin, show=DEBUG):
    def arousalScore(t):
        before = pitches[(times >= t - TUNE_SCOPE / 2) & (times <= t)]
        after = pitches[(times >= t) & (times <= t + TUNE_SCOPE / 2)]
        before = (librosa.hz_to_midi(before + 0.1) * 6 / 12).astype(int)
        after = (librosa.hz_to_midi(after + 0.1) * 6 / 12).astype(int)
        score = np.sum(after) - np.sum(before)
        return score / len(before)

    mask = (times >= time - window / 2) & (times <= time + window / 2)
    scores = [arousalScore(t) for t in times[mask]]
    point = times[mask][np.argmax(scores)] if begin else times[mask][np.argmin(scores)]
    if show:
        logger.debug(
            f"point={point} times={times[mask][0]}~{times[mask][-1]} window={window}"
        )
        plt.plot(times[mask], pitches[mask], label="pitch")
        plt.plot(times[mask], scores, label="score")
        plt.scatter(point, np.max(scores) if begin else np.min(scores))
        plt.xlabel("time/s")
        plt.ylabel("freq/Hz")
        plt.legend()
        plt.show()
    return point


def tuneIntervals(mirexFmt, mels_f, chorusDur, window):
    mirexFmt = removeNumber(mirexFmt)
    mirexFmt = mergeIntervals(mirexFmt)
    logger.debug(f"tune interval=\n{mirexLines(mirexFmt)}")
    dur = mirexFmt[0][-1][1]
    intvs = filterIntvs(mirexFmt, fun=CLF_TARGET_LABEL)
    tuneIntvs = []
    times, pitches = mels_f
    for intv in intvs:
        begin = arousalPoint(intv[0], times, pitches, window, True)
        end = arousalPoint(intv[1], times, pitches, window, False)
        end = min(dur, max(end, begin + chorusDur))
        if end - begin > MINIMUM_CHORUS_DUR:
            tuneIntvs.append((begin, end))
    return multiChorusSections(tuneIntvs, dur)
