import click
import librosa
import numpy as np

from main import extract


@click.command()
@click.argument("audiofile", nargs=1, type=click.Path(exists=True))
@click.argument("output", nargs=1, type=click.Path())
def main(audiofile, output):
    fs = [audiofile]
    highlightList = list(
        extract(fs, length=30, save_score=False, save_thumbnail=False, save_wav=False)
    )
    begin, end = highlightList[0]
    dur = librosa.get_duration(filename=audiofile)
    intervals = [
        (0, begin),
        (begin, end),
        (end, dur),
    ]
    labels = [
        "others",
        "chorus",
        "others",
    ]

    contents = np.array(
        [(x[0], x[1], y) for x, y in zip(intervals, labels)], np.dtype("f, f, U16")
    )
    np.savetxt(output, contents, fmt=["%.2f", "%.2f", "%s"], delimiter="\t")


if __name__ == "__main__":
    main()
